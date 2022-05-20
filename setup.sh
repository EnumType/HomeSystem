clear &&
sudo apt update && sudo apt upgrade &&
sudo apt install default-jdk default-jre git python3 python3-pip apache2 screen
sudo pip3 install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html &&
sudo pip3 install pandas &&
chmod +x ./start.sh