curl -sSL https://install.python-poetry.org | python3 -
sh -c 'echo "export PATH=\"$HOME/.local/bin:\$PATH\"" >> /etc/bash.bashrc'
git clone https://MikolajSzawerda:<TOKEN>@github.com/MikolajSzawerda/musical-generative-models-conditioning.git
cd musical-generative-models-conditioning
make vc-env-audiocraft
