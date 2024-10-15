package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	_ "github.com/go-sql-driver/mysql"
	"github.com/sashabaranov/go-openai"
)

type Feed struct {
	Entries []Entry `xml:"entry"`
}

type Entry struct {
	ID                  string   `xml:"id"`
	Title               string   `xml:"title"`
	Summary             string   `xml:"summary"`
	Authors             []Author `xml:"author"`
	Published           string   `xml:"published"`
	Categories          []string `xml:"category"`
	PDFLink             []Link   `xml:"link"`
	GPTSummary          string
	GPTNewContributions string
	GPTTags             []string
}

type Author struct {
	Name string `xml:"name"`
}

type Link struct {
	Href string `xml:"href,attr"`
	Rel  string `xml:"rel,attr"`
}

type Query struct {
	ID    string `json:"id"`
	Query string `json:"query"`
}

type Queries struct {
	Queries []Query `json:"queries"`
}
type GPTResponse struct {
	Summary          string   `json:"summary"`
	NewContributions string   `json:"novel_contributions"`
	Tags             []string `json:"tags"`
}

func constructArxivAPIURL(query string) string {
	baseURL := "http://export.arxiv.org/api/query"
	queryParam := "?search_query=" + query + ""
	return baseURL + queryParam
}

func callArxivAPI(queryURL string) ([]byte, error) {
	resp, err := http.Get(queryURL)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	return body, nil
}

func parseArxivXMLResponse(data []byte) (*Feed, error) {
	var feed Feed
	err := xml.Unmarshal(data, &feed)
	if err != nil {
		return nil, err
	}
	return &feed, nil
}

func extractCategories(entry Entry) string {
	var categories []string
	for _, category := range entry.Categories {
		categories = append(categories, category)
	}
	return strings.Join(categories, ",")
}

func extractAuthors(entry Entry) string {
	var authors []string
	for _, author := range entry.Authors {
		authors = append(authors, author.Name)
	}
	return strings.Join(authors, ", ")
}

func extractPDFURL(entry Entry) string {
	for _, link := range entry.PDFLink {
		if link.Rel == "related" && strings.Contains(link.Href, "pdf") {
			return link.Href
		}
	}
	return ""
}

func parsePublishedDate(published string) (time.Time, error) {
	return time.Parse(time.RFC3339, published)
}

func callArxivStub(filename string) ([]byte, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	body, err := io.ReadAll(file)
	if err != nil {
		return nil, err
	}

	return body, nil
}

func saveToDatabase(db *sql.DB, entry Entry, queryID string) error {
	categories := extractCategories(entry)
	authors := extractAuthors(entry)
	pdfURL := extractPDFURL(entry)
	publicationDate, err := parsePublishedDate(entry.Published)
	if err != nil {
		return err
	}

	query := `INSERT INTO papers (paper_id, title, summary, categories, pdf_url, authors, publication_date, query_id) 
			  VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
	_, err = db.Exec(query, entry.ID, entry.Title, entry.Summary, categories, pdfURL, authors, publicationDate, queryID)
	return err
}

func readQueriesFromJSON(filename string) ([]Query, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	bytes, err := io.ReadAll(file)
	if err != nil {
		return nil, err
	}

	var queries Queries
	err = json.Unmarshal(bytes, &queries)
	if err != nil {
		return nil, err
	}

	return queries.Queries, nil
}

func getExistingPaperIDs(db *sql.DB, paperIDs []string) (map[string]bool, error) {
	existingPapers := make(map[string]bool)
	query := `SELECT paper_id FROM papers WHERE paper_id IN (?` + strings.Repeat(",?", len(paperIDs)-1) + `)`
	args := make([]interface{}, len(paperIDs))
	for i, id := range paperIDs {
		args[i] = id
	}

	rows, err := db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var paperID string
	for rows.Next() {
		if err := rows.Scan(&paperID); err != nil {
			return nil, err
		}
		existingPapers[paperID] = true
	}
	return existingPapers, nil
}

func saveToDatabaseBatch(db *sql.DB, entries []Entry, queryID string) error {
	if len(entries) == 0 {
		return nil
	}

	query := `INSERT INTO papers (paper_id, title, summary, categories, pdf_url, authors, publication_date, query_id, gpt_summary, gpt_contributions, gpt_tags) 
			  VALUES `
	values := make([]interface{}, 0, len(entries)*10)

	for i, entry := range entries {
		if i > 0 {
			query += ","
		}
		query += "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
		categories := extractCategories(entry)
		authors := extractAuthors(entry)
		pdfURL := extractPDFURL(entry)
		publicationDate, _ := parsePublishedDate(entry.Published)

		gptSummary := entry.GPTSummary
		gptContributions := entry.GPTNewContributions
		gptTags := strings.Join(entry.GPTTags, ", ")

		values = append(values, entry.ID, entry.Title, entry.Summary, categories, pdfURL, authors, publicationDate, queryID, gptSummary, gptContributions, gptTags)
	}

	_, err := db.Exec(query, values...)
	return err
}

func processQuery(db *sql.DB, query Query) {
	fmt.Printf("Processing query ID: %s with query: %s\n", query.ID, query.Query)
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("Please set your OPENAI_API_KEY environment variable")
	}

	client := openai.NewClient(apiKey)
	url := constructArxivAPIURL(query.Query)
	// data, err := callArxivStub("stub.xml")
	data, err := callArxivAPI(url)
	if err != nil {
		log.Fatalf("Failed to call ArXiv API: %v", err)
	}

	feed, err := parseArxivXMLResponse(data)
	if err != nil {
		log.Fatalf("Failed to parse XML response: %v", err)
	}
	var paperIDs []string
	for _, entry := range feed.Entries {
		paperIDs = append(paperIDs, entry.ID)
	}
	if len(paperIDs) == 0 {
		log.Printf("No entries ofr query: %s", query.ID)
		return
	}
	existingPapers, err := getExistingPaperIDs(db, paperIDs)
	if err != nil {
		log.Fatalf("Failed to check existing papers for query ID %s: %v", query.ID, err)
	}

	var newEntries []Entry
	enrichedEntries := make(chan Entry)
	var wg sync.WaitGroup
	for _, entry := range feed.Entries {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, exists := existingPapers[entry.ID]; exists {
				return
			}
			log.Printf("Call gpt for %s", entry.ID)
			gptResponse, err := getGPTInfoForPaper(client, entry.Summary)
			if err != nil {
				log.Printf("Failed to call GPT for paper %s: %v", entry.ID, err)
				return
			}
			entry.GPTSummary = gptResponse.Summary
			entry.GPTNewContributions = gptResponse.NewContributions
			entry.GPTTags = gptResponse.Tags
			enrichedEntries <- entry
		}()

	}
	go func() {
		wg.Wait()
		close(enrichedEntries)
	}()
	for entry := range enrichedEntries {
		newEntries = append(newEntries, entry)
	}

	err = saveToDatabaseBatch(db, newEntries, query.ID)
	if err != nil {
		log.Printf("Failed to save entries to DB for query ID %s: %v", query.ID, err)
	} else {
		fmt.Printf("Saved %d new papers for query ID %s to the database.\n", len(newEntries), query.ID)
	}
	if len(newEntries) == 0 {
		return
	}
	fileName := fmt.Sprintf("%3d_%s.md", time.Now().Nanosecond()/1e6, query.ID)
	err = saveEntriesToMarkdown(newEntries, fmt.Sprintf("summary/%s/%s", time.Now().Format("02-01"), fileName))
	if err != nil {
		log.Print("Failed to save entries to md file")
	}
}
func saveEntriesToMarkdown(entries []Entry, fileName string) error {
	var sb strings.Builder

	// Create a Markdown header
	sb.WriteString("# Research Papers Summary\n\n")

	for _, entry := range entries {
		// Write the title as a markdown header
		title := strings.ReplaceAll(entry.Title, "\n", "")
		title = strings.ReplaceAll(title, "\t", "")
		sb.WriteString(fmt.Sprintf("## %s\n\n", title))

		// Add details of the paper
		sb.WriteString(fmt.Sprintf("- **ID**: %s\n", entry.ID))
		sb.WriteString(fmt.Sprintf("- **Published**: %s\n", entry.Published))

		// List authors
		sb.WriteString("- **Authors**: ")
		authorNames := []string{}
		for _, author := range entry.Authors {
			authorNames = append(authorNames, author.Name)
		}
		sb.WriteString(strings.Join(authorNames, ", "))
		sb.WriteString("\n")

		// List categories
		sb.WriteString("- **Categories**: ")
		sb.WriteString(strings.Join(entry.Categories, ", "))
		sb.WriteString("\n")

		// Add GPT-generated summary and contributions
		sb.WriteString(fmt.Sprintf("\n### GPT Summary\n%s\n", entry.GPTSummary))
		sb.WriteString(fmt.Sprintf("\n### New Contributions\n%s\n", entry.GPTNewContributions))

		// Add GPT tags
		sb.WriteString("\n### Tags\n")
		sb.WriteString(strings.Join(entry.GPTTags, ", "))
		sb.WriteString("\n")

		// Add a link to the PDF if available
		if len(entry.PDFLink) > 0 {
			sb.WriteString("\n### PDF Link\n")
			sb.WriteString(fmt.Sprintf("[Link](%s)\n", entry.PDFLink[0].Href))
		}
		sb.WriteString("\n---\n\n")
	}
	err := os.MkdirAll(filepath.Dir(fileName), os.ModePerm)
	if err != nil {
		return err
	}
	file, err := os.Create(fileName)
	if err != nil {
		return err
	}
	defer file.Close()

	_, err = file.WriteString(sb.String())
	if err != nil {
		return err
	}

	return nil
}
func escapeStringForJSON(input string) (string, error) {
	escapedBytes, err := json.Marshal(input)
	if err != nil {
		return "", err
	}
	escapedString := string(escapedBytes)
	return escapedString[1 : len(escapedString)-1], nil
}
func getGPTInfoForPaper(client *openai.Client, abstract string) (GPTResponse, error) {
	ctx := context.Background()
	friendlyAbstract, err := escapeStringForJSON(abstract)
	if err != nil {
		return GPTResponse{}, err
	}
	resp, err := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: openai.GPT4oMini,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: "You are an AI specialized in scientific research summarization. You will be provided with the abstract of a research paper. Your task is to: 1. Generate a concise summary of the paper (1-2 sentences). 2. Clearly describe the novel contributions or findings the paper introduces. 3. Provide a list of relevant tags (5-10) that are specific and informative. The tags should focus on the paper's key topics and areas, particularly in the context of conditioning of musical generative models or related fields. Avoid general terms like 'AI' or 'machine learning' or any general tag that is obvious. Your response must always be in valid JSON format with the following structure: { \"summary\": \"Your concise summary here.\", \"novel_contributions\": \"Description of what new contributions the paper introduces.\", \"tags\": [\"specific_tag1\", \"specific_tag2\", \"specific_tag3\", \"...\"] } Ensure that all fields are filled accurately and that the response strictly adheres to the above structure.",
			},
			{
				Role:    openai.ChatMessageRoleUser,
				Content: friendlyAbstract,
			},
		},
		Temperature: 0.7,
		MaxTokens:   300,
	})

	if err != nil {
		return GPTResponse{}, err
	}

	var gptResponse GPTResponse
	err = json.Unmarshal([]byte(resp.Choices[0].Message.Content), &gptResponse)
	if err != nil {
		return GPTResponse{}, err
	}

	return gptResponse, nil
}
func fetchEntriesFromDB(db *sql.DB) ([]Entry, error) {
	query := `SELECT paper_id, title, summary, authors, publication_date, categories, pdf_url, gpt_summary, gpt_contributions, gpt_tags FROM papers ORDER BY publication_date desc`
	rows, err := db.Query(query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var entries []Entry

	for rows.Next() {
		var entry Entry
		var authors, categories, pdfLinks, tags string

		// Scan each row into the corresponding Entry fields
		err := rows.Scan(&entry.ID, &entry.Title, &entry.Summary, &authors, &entry.Published, &categories, &pdfLinks, &entry.GPTSummary, &entry.GPTNewContributions, &tags)
		if err != nil {
			return nil, err
		}

		// Parse comma-separated authors
		authorNames := strings.Split(authors, ",")
		for _, authorName := range authorNames {
			entry.Authors = append(entry.Authors, Author{Name: strings.TrimSpace(authorName)})
		}

		// Parse comma-separated categories
		entry.Categories = strings.Split(categories, ",")

		// Parse comma-separated PDF links
		linkParts := strings.Split(pdfLinks, ",")
		for _, link := range linkParts {
			entry.PDFLink = append(entry.PDFLink, Link{Href: strings.TrimSpace(link), Rel: "alternate"}) // Assuming "Rel" is always "alternate"
		}

		// Parse comma-separated GPT tags
		entry.GPTTags = strings.Split(tags, ",")

		entries = append(entries, entry)
	}

	// Check for errors during iteration
	if err = rows.Err(); err != nil {
		return nil, err
	}

	return entries, nil
}
func createTablesIfNotExist(db *sql.DB) error {
	createTableSQL := `
    CREATE TABLE IF NOT EXISTS papers (
		paper_id VARCHAR(255) PRIMARY KEY,
		title TEXT,
		summary TEXT,
		categories TEXT,
		pdf_url TEXT,
		authors TEXT,
		publication_date DATETIME,
		query_id VARCHAR(255),
		gpt_summary TEXT,
		gpt_contributions TEXT,
		gpt_tags TEXT
    );
    `
	_, err := db.Exec(createTableSQL)
	return err
}
func main() {

	dbUser := "myuser"
	dbPassword := "mypassword"
	dbName := "mydb"
	dbHost := "127.0.0.1"
	dbPort := "3306"

	dsn := fmt.Sprintf("%s:%s@tcp(%s:%s)/%s", dbUser, dbPassword, dbHost, dbPort, dbName)

	db, err := sql.Open("mysql", dsn)
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()
	err = createTablesIfNotExist(db)
	if err != nil {
		log.Fatal(err)
	}

	queries, err := readQueriesFromJSON("queries.json")
	if err != nil {
		log.Fatalf("Failed to read queries from JSON file: %v", err)
	}

	for _, query := range queries {
		processQuery(db, query)
	}

	fmt.Println("All queries processed.")
	entries, err := fetchEntriesFromDB(db)
	if err != nil {
		log.Fatal("Failed to retrieve data")
	}
	saveEntriesToMarkdown(entries, "autoresearch.md")
	fmt.Println("All rows saved to markdown")
}
