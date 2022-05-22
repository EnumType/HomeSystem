## Getting Started

These instructions will get you a copy of the project up and running on your local machine to deploy the HomeSystem server.

### Prerequisites

Before you download the HomeSystem you should check if you have the following things running on your machine:
- Any linux 64bit system
- Latest version of git: `sudo apt install git`

### Installing

Create a new directory and move into it. Then execute `git clone https://github.com/TheJavaException/HomeSystem` and move into *HomeSystem/*. Copy or move the *Server.jar*, *start.sh* and *setup.sh* to your created directory and execute `chmod +x setup.sh`. After that you need to run `chmod +x ./setup.sh` and `./setup.sh` to install all required packages. Now, you just have to execute `./start.sh` and a screen with the HomeSystem-Server should be running. You can enter this screen with `screen -r HomeSystem`.  
Now you are free to remove the cloned directory *HomeSystem/* to save some disk space.

## Configuration

### Config.cfg

- `XmlRpc-Address` The address of your HomeMatic CCU
- `XmlRpc-Port` The port of the XmlRpc server of your CCU (default port is 2001)
- `HmIP` The port of the HomeMaticIP XmlRpc server (default port is 2010)
- `Resources-Dir` The directory where all resources are located (e.g. the WebSocket keystore.jks)
- `WebSocket-Http` The port on which the HTTP WebSocket is running (default port is 8000. If you change it you should change the port in the PHP files too!)
- `WebSocket-Https` The port on which the HTTPS WebSocket is running (default port is 8000. If you change it you should change the port in the PHP files too!)
- `WebSocket-Keystore` The name of your keystore which is located in the *resources* directory (Please generate a keystore.jks! Otherwise, the WebSocket server will not be started!)
- `WebSocket-KeystorePassword` The password of your keystore file
- `BrightnessSensor` The address of the Brightness sensor used for AI. Change to *none* if you don't have a HomeMatic Brightness Sensor.
- `BrightnessSensorHmIP` If the Brightness sensor is a HomeMaticIP sensor
- `AI-Interval` Interval for AIData saving and AI prediction in minutes. (See [AI](#AI))
- `HashSalt` Charset which will be used to calculate the password hash. If you want to change it, change it in the `index.html` as well! 

### User.data

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
- `Address` The address of the device. You can get it from your HomeMatic WebUI.
- `HmIP` If your device is a HomeMaticIP device turn it on true
- `AIData` Should the system save data from this device for AI learning. See [AI](#AI)
- `AIControl` Should the AI control this system. See [AI](#AI)
- `Permission` The permission for this room. See [Permissions.yml](#Permissions.yml)

### Permissions.yml

In the *Permissions.yml* you can add permissions to each user. These users can access to the rooms which also have this permission.

## The System

### AI

The *HomeSystem* contains an AI to predict the state of your devices based on time, date and brightness(weather). If you want that the AI controls your devices turn *AIData* on *true* and let the system save the data from your device. After one or two months you can turn the *AIControl* on *true*, and the AI will predict which state the device should have. You can turn *AIData* back to *false*, but it is recommended to leave it on *true* because then the system will continue saving data. The predictions will be more correctly with more device (training) data. (In the latest version 1.0.8 the AI has been implemented. The AI is in a BETA-Mode and can produce errors!)

Supported devices:
- ROLL -> can only be predicted and controlled in state *up* or *down*
- LAMP -> can be predicted and controlled in state *on* or *off*

### Website

The website files can be extracted with the command `extract website`. They will be written into the directory *HTTP*. After that you can move them into your webserver (Apache2) directory and can run them in your browser.   
HTTPS is necessary because the inserted password will be sent without any end-to-end encryption.

### Plugin system

The system has an implemented plugin system, so you can write your own plugins to bring new features to the system. See more at [Plugin Interface](Plugin Interface.md).

## Libraries and Languages

- [Java](https://www.oracle.com/de/java/) - Main system
- [Python](https://www.python.org/) - AI language
- [Pytorch](https://www.pytorch.org/) - Library for AI stuff
- [Pandas](https://pandas.pydata.org/) - Library to fetch data to the AI stuff
- [JavaScript](https://en.wikipedia.org/wiki/JavaScript) - Website stuff
- [HTML](https://en.wikipedia.org/wiki/HTML) - More website stuff
- [CSS](https://en.wikipedia.org/wiki/Cascading_Style_Sheets) - Style for the website stuff
- [Apache XML-RPC](https://ws.apache.org/xmlrpc/) - Communication to the HomeMatic CCU
- [WS-Common Utilities](https://ws.apache.org/commons/util/) - API required for XML-RPC communication
- [Jetty-All](https://www.eclipse.org/jetty/) - WebSocket stuff
- [JSON-Simple](https://code.google.com/archive/p/json-simple/) - JSON stuff
- [Snake YAML](https://bitbucket.org/asomov/snakeyaml/src/master/) - YAML configuration stuff

## Authors

- **[EnumType](https://github.com/EnumType)** - *Idea and Coding*
- **[Letsmoe](https://github.com/Letsmoe)** - *Pretty good help with these AI things*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Version log

- See [Releases](https://github.com/TheJavaException/HomeSystem/releases)