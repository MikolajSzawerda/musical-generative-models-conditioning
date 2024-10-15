sudo apt-get update
sudo apt-get install -y locales
sudo locale-gen en_US.UTF-8
sudo update-locale LANG=en_US.UTF-8
export LANG=en_US.UTF-8
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python3-openssl \
    git

#pyenv installation
curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"
      [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
      eval "$(pyenv init -)"' >> ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"
      [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
      eval "$(pyenv init -)"' >> ~/.bash_profile
source ~/.bashrc
pyenv install 3.9

#just installation
mkdir -p ~/bin
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/bin
export PATH="$PATH:$HOME/bin"
just --help

curl -sSL https://install.python-poetry.org | python3 -
sh -c 'echo "export PATH=\"$HOME/.local/bin:\$PATH\"" >> /etc/bash.bashrc'
git clone https://MikolajSzawerda:<TOKEN>@github.com/MikolajSzawerda/musical-generative-models-conditioning.git
cd musical-generative-models-conditioning
just prepare-audiocraft
