# HomeSystem
The HomeSystem is a Java based system which allows you to controll your HomeMatic(IP) devices over the implemented WebSocket server.

## Gettin Started
These instructions will get you a copy of the project up and running on your local machine to deploy the HomeSystem server.

### Prerequisites
Before you download the HomeSystem you should install the following sorftware to run this system:
- Oracle-JDK `sudo apt install default-jdk`
- Oracle-JRE `sudo apt install default-jre`
- Git `sudo apt install git`
- Python3 `sudo apt install python3`
- Pip3 `sudo apt install python3-pip`
- Pytorch `sudo pip3 install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html` (Details on https://pytorch.org)
- Pandas `sudo pip3 install pandas`
- Apache2 `sudo apt install apache2`
- Screen `sudo apt install screen`

### Installing
Create a new directory and move in it. Then execute `git clone https://github.com/TheJavaException/HomeSystem` and move into *HomeSystem/*. Copy the *Server.jar* and the *start.sh* to your created directory and execute `chmod +x start.sh`. Finally you just have to execute `./start.sh` and a screen with the HomeSystem-Server should be running. You can enter this screen with `screen -r HomeSystem`.

## Configuration
After the installing you should change a few parameters in the *config.txt* to keep the system running.

### Config.txt
- `XmlRpc-Address` The address of your HomeMatic CCU
- `XmlRpc-Port` The port of the XmlRpc server of your CCU (default port is 2001)
- `HmIP` The port of the HomeMaticIP XmlRpc server (default port is 2010)
- `Server-Port` Port on which the Java ServerSocket is running
- `Resources-Dir` The directory where all ressources are located (e.g. the WebSocket keystore.jks)
- `WebSocket-Http` The port on which the HTTP WebSocket is running (default port is 8000. If you change it you should change the port in the PHP files too!)
- `WebSocket-Https` The port on which the HTTPS WebSocket is running (default port is 8000. If you change it you should change the port in the PHP files too!)
- `WebSocket-Keystore` The name of your keystore which is located in the *resources* directory (Please generate a keystore.jks! Otherwise the WebSocket server will not be started!)
- `WebSocket-KeystorePassword` The password of your keystore file
- `BrightnessSensor` The address of the Brightness sensor used for AI. Change to *none* if you don't have a HomeMatic Brightness Sensor
- `BrightnessSensorHmIP` If the Brightness sensor is a HomeMaticIP sensor
- `AI-Interval` Interval for AIData saving and AI prediction in minutes. (See [AI](#AI))

### Data.txt
The login data for the users is located in the *User-Data* directory.
`Name` The username for this account.  
`Password` The password hash for the user (Generate your password e.g. on [Passwordgenerator.net](https://passwordsgenerator.net/))

```
Username: user1
Password: hashed passoword
Username: user2
Password: hashed passoword
```

### Rooms.yml
In the *Rooms.yml* you can define which rooms are existing and which devices are in them.
- `Type` The type of the Device (*ROLL* and *LAMP* available)
- `Address` The address of the device. You can get it from your HomeMatic WebUI
- `HmIP` If your device is a HomeMaticIP device turn it on true
- `AIData` Should the system save data from this device for AI lerning. See [AI](#AI)
- `AIControl` Should the AI control this system. See [AI](#AI)
- `Permission` The permission for this room. See [Permissions.yml](#Permissions.yml)

### Permissions.yml
In the *Permissions.yml* you can add permissions to each user. This users can access to the rooms which also have this permission

## The System

### AI
The *HomeSystem* contains an AI to predict the state of your devices based on time, date and brightness(weather). If you want that the AI controls your devices turn *AIData* on *true* and let the system save the data from your device. After one or two month you can turn the *AIControl* on *true* and the AI will predict which state the device should have. You can turn *AIData* back to *false* but it is recommended to leave it on *true* because then the system will continue saving data. The predictions will be more corectly with more device (training) data. (In the latest version 1.0.8 the AI has been implemented. The AI is in a BETA-Mode and can produce errors!)

Supported devices:
- ROLL -> can only be predicted and controlled in state *up* or *down*
- LAMP -> can be predicted and controlled in state *on* or *off*

### Website
The website files can be extracted with the command `extract website`. They will be written into the directory *HTTP*. After that you can move them into your webserver (Apache2) directory and can run them in your browser. (In the latest version 1.0.8 the connection will be over http by default!).

If you want to connect over HTTPS change `var ws = new WebSocket("ws://" + ip + ":8000/")` in *index.php* and *home.php* to `var ws = new WebSocket("wss://" + ip + ":8001/")` where the *8001* is the Https port of your WebSocket server in the System. (See [Config](#Config.txt))  
HTTPS is not necessary because the inserted password will be salted and hashed by using SHA-512, and the hash will be sent. 

## Built With
- [Java](https://www.oracle.com/de/java/) - Language used for the main system
- [Python](https://www.python.org/) - Language used for the AI
- [Pytorch](https://www.pytorch.org/) - Libary for Machine learning used for the AI
- [Pandas](https://pandas.pydata.org/) - Libary for Data Analysis used to extract saved data to the AI
- [JavaScript](https://en.wikipedia.org/wiki/JavaScript) - Language used for the WebSocket WebBrowser communication
- [HTML](https://en.wikipedia.org/wiki/HTML) - Language used for the Website files
- [CSS](https://en.wikipedia.org/wiki/Cascading_Style_Sheets) - Style Language used for the Style of the Website
- [Apache XML-RPC](https://ws.apache.org/xmlrpc/) - Apache XML-RPC API to communicate with the HomeMatic XML-RPC Server
- [Image4j](http://image4j.sourceforge.net/)
- [Jetty-All](http://image4j.sourceforge.net/)
- [JSON-Simple](https://code.google.com/archive/p/json-simple/)
- [Snake YAML](https://bitbucket.org/asomov/snakeyaml/src/master/)
- [WS-Commons-Util](https://mvnrepository.com/artifact/org.apache.ws.commons.util/ws-commons-util/1.0.2)

## Authors
- **EnumType** - *Idea and Coding*

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Version log
- See [Releases](https://github.com/TheJavaException/HomeSystem/releases)
