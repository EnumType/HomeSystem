package net.enumtype.homesystem.utils;

import net.enumtype.homesystem.Main;
import net.enumtype.homesystem.rooms.Device;

import java.io.*;

public class Data {

	private final Log log;
	private final Device aiBrightSensor;
	private String xmlRpcAddress;
	private String wsKeystore;
	private String wsKeystorePassword;
	private String resourcesDir;
	private String version;
	private String hashSalt;
	private int xmlRpcPort;
	private int hmIpPort;
	private int wsPort;
	private int wssPort;
	private int aiInterval;
	private boolean printStackTraces;

	public Data() {
		this.aiBrightSensor = new Device();
		this.log = Main.getLog();
		load();
	}

	public void load() {
		loadXmlRpcData();
	}

	public void loadXmlRpcData() {
		try {
			File config = new File("config.txt");

			if(config.exists()) {
				log.write("Getting Configs...", true, false);
				BufferedReader reader = new BufferedReader(new FileReader(config));
				String line;

				while((line = reader.readLine()) != null) {
					final String [] args = line.split(" ");
					if(args.length < 1) continue;

					switch (args[0].replaceAll(":", "")) {
						case "XmlRpc-Address":
							xmlRpcAddress = args[1];
							break;
						case "XmlRpc-Port":
							xmlRpcPort = Integer.parseInt(args[1]);
							break;
						case "HmIP-Port":
							hmIpPort = Integer.parseInt(args[1]);
							break;
						case "Resources-Dir":
							resourcesDir = args[1];
							break;
						case "WebSocket-Http":
							wsPort = Integer.parseInt(args[1]);
							break;
						case "WebSocket-Https":
							wssPort = Integer.parseInt(args[1]);
							break;
						case "WebSocket-Keystore":
							wsKeystore = args[1];
							break;
						case "WebSocket-KeystorePassword":
							wsKeystorePassword = args[1];
							break;
						case "BrightnessSensor":
							aiBrightSensor.setAddress(args[1]);
							break;
						case "BrightnessSensorHmIP":
							aiBrightSensor.setHmIp(Boolean.parseBoolean(args[1]));
							break;
						case "AI-Interval":
							aiInterval = Integer.parseInt(args[1]);
							break;
						case "HashSalt":
							hashSalt = args[1];
							break;
						case "PrintStackTraces":
							printStackTraces = Boolean.parseBoolean(args[1]);
						default:
							log.write("Unknown config parameter '" + line + "'!", false, false);
					}
				}

				reader.close();
			}else {
				log.write("Creating Configs...", true, false);
				InputStream stream = Main.class.getResourceAsStream("/config.txt");
				BufferedReader in = new BufferedReader(new InputStreamReader(stream));
				BufferedWriter out = new BufferedWriter(new FileWriter(config));

				String line;
				while((line = in.readLine()) != null) {
					out.write(line + "\r\n");
				}

				in.close();
				out.close();
				loadXmlRpcData();
			}
		}catch(IOException | NumberFormatException e) {
			log.writeError(e);
		}
	}

	public void setVersion(String version) {
		this.version = version;
	}

	public String getVersion() {return version;}
	public String getWsKeystore() {return wsKeystore;}
	public String getWsKeystorePassword() {return wsKeystorePassword;}
	public String getResourcesDir() {return resourcesDir;}
	public String getXmlRpcAddress() {return xmlRpcAddress;}
	public String getHashSalt() {return hashSalt;}
	public Device getAiBrightSensor() {return aiBrightSensor;}
	public int getWsPort() {return wsPort;}
	public int getWssPort() {return wssPort;}
	public int getAiInterval() {return aiInterval;}
	public int getXmlRpcPort() {return xmlRpcPort;}
	public int getHmIpPort() {return hmIpPort;}
	public boolean printStackTraces() {return printStackTraces;} //for debugging only

}
