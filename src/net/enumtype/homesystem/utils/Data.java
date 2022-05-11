package net.enumtype.homesystem.utils;

import net.enumtype.homesystem.main.Main;
import net.enumtype.homesystem.xmlrpc.Device;

import java.io.*;

public class Data {

	private final Log log;
	private final Device aiBrightSensor;
	private String xmlRpcAddress;
	private String wsKeystore;
	private String wsKeystorePassword;
	private String resourcesDir;
	private String version;
	private int xmlRpcPort;
	private int hmIpPort;
	private int wsPort;
	private int wssPort;
	private int aiInterval;

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
				log.write(Methods.createPrefix() + "Getting Configs...", true);
				BufferedReader reader = new BufferedReader(new FileReader(config));
				String line;

				while((line = reader.readLine()) != null) {
					if(line.startsWith("XmlRpc-Address")) {
						line = line.replace("XmlRpc-Address: ", "");
						xmlRpcAddress = line;
					}else if(line.startsWith("XmlRpc-Port")) {
						line = line.replace("XmlRpc-Port: ", "");
						xmlRpcPort = Integer.parseInt(line);
					}else if(line.startsWith("HmIP-Port")) {
						line = line.replace("HmIP-Port: ", "");
						hmIpPort = Integer.parseInt(line);
					}else if(line.startsWith("Resources-Dir")) {
						line = line.replace("Resources-Dir: ", "");
						resourcesDir = line;
					}else if(line.startsWith("WebSocket-Http") && !line.startsWith("WebSocket-Https")) {
						line = line.replace("WebSocket-Http: ", "");
						wsPort = Integer.parseInt(line);
					}else if(line.startsWith("WebSocket-Https")) {
						line = line.replace("WebSocket-Https: ", "");
						wssPort = Integer.parseInt(line);
					}else if(line.startsWith("WebSocket-Keystore") && !line.startsWith("WebSocket-KeystorePassword")) {
						line = line.replace("WebSocket-Keystore: ", "");
						wsKeystore = line;
					}else if(line.startsWith("WebSocket-KeystorePassword")) {
						line = line.replace("WebSocket-KeystorePassword: ", "");
						wsKeystorePassword = line;
					}else if(line.startsWith("BrightnessSensor") && !line.startsWith("BrightnessSensorHmIP")) {
						line = line.replace("BrightnessSensor: ", "");
						aiBrightSensor.setAddress(line);
					}else if(line.startsWith("BrightnessSensorHmIP")) {
						line = line.replace("BrightnessSensorHmIP: ", "");
						aiBrightSensor.setHmIp(Boolean.parseBoolean(line));
					}else if(line.startsWith("AI-Interval")) {
						line = line.replace("AI-Interval: ", "");
						aiInterval = Integer.parseInt(line);
					}
				}

				reader.close();
			}else {
				log.write(Methods.createPrefix() + "Creating Configs...", true);
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
			e.printStackTrace();
			log.write(Methods.createPrefix() + "Error in Data(99): " + e.getMessage(), false);
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
	public Device getAiBrightSensor() {return aiBrightSensor;}
	public int getWsPort() {return wsPort;}
	public int getWssPort() {return wssPort;}
	public int getAiInterval() {return aiInterval;}
	public int getXmlRpcPort() {return xmlRpcPort;}
	public int getHmIpPort() {return hmIpPort;}

}
