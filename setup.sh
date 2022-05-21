clear &&
sudo apt update && sudo apt upgrade &&
sudo apt install default-jdk default-jre git python3 python3-pip apache2 screen &&
sudo pip3 install pandas matplotlib &&
sudo pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu &&
chmod +x ./start.sh