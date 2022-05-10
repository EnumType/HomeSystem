package net.enumtype.homesystem.utils;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.net.URLConnection;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;

import javax.imageio.ImageIO;
import javax.net.ssl.HttpsURLConnection;

import net.enumtype.homesystem.main.Main;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import net.sf.image4j.codec.ico.ICOEncoder;

public class Methods {
	
	public static String createPrefix() {
		System.out.println();
		return "[" + new SimpleDateFormat("HH:mm:ss").format(new Date()) + "] ";
	}
	
	public static String getDate() {
		return new SimpleDateFormat("yyyy-MM-dd").format(new Date());
	}
	
	public static long getUnixTime() {
		return System.currentTimeMillis() / 1000;
	}
	
	public static void extractWebsite() throws IOException {
		Log.write(createPrefix() + "Extracting Website-Data...", true);
		
		File path = new File("HTTP");
		File faviconICO = new File(path + "//favicon.ico");
		File faviconPNG = new File(path + "//favicon.png");
		File index = new File(path + "//index.php");
		File home = new File(path + "//home.php");
		File style = new File(path + "//style.css");
		
		if(!path.exists()) if(!path.mkdir()) throw new IOException("Cannot create directory!");
		if(faviconICO.exists()) if(!faviconICO.delete()) throw new IOException("Cannot delete file!");
		if(faviconPNG.exists()) if(!faviconPNG.delete()) throw new IOException("Cannot delete file!");
		if(index.exists()) if(!index.delete()) throw new IOException("Cannot delete file!");
		if(home.exists()) if(!home.delete()) throw new IOException("Cannot delete file!");
		if(style.exists()) if(!style.delete()) throw new IOException("Cannot delete file!");
		
		writeResources(faviconICO, "/HTML/favicon.png", true, "ico");
		writeResources(faviconPNG, "/HTML/favicon.png", true, "png");
		writeResources(index, "/HTML/index.php", false, "php");
		writeResources(home, "/HTML/home.php", false, "php");
		writeResources(style, "/HTML/style.css", false, "css");
		
		Log.write(createPrefix() + "Extracted Website-Data", true);
	}
	
	public static void writeResources(File file, String resource, boolean image, String format) {
		try {
			if(image) {
				final BufferedImage bImage = ImageIO.read(Main.class.getResourceAsStream(resource));
				if(format.equals("ico")) {
					ICOEncoder.write(bImage, file);
				}else {
					ImageIO.write(bImage, format, file);
				}
			}else {
				InputStream stream = Main.class.getResourceAsStream(resource);
				BufferedReader in = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8));
				BufferedWriter out = new BufferedWriter(new FileWriter(file));
				
				String line;
				while((line = in.readLine()) != null) {
					out.write(line + "\r\n");
				}
				
				in.close();
				out.close();
			}
		}catch(IOException e) {
			e.printStackTrace();
			Log.write(createPrefix() + "Error in Methods(117): " + e.getMessage(), false);
		}
	}
	
	public static void startVersionChecking() {
		int wait = 2;
		new Thread(() -> {
			try {
				while(true) {
					String url = "https://api.github.com/repos/TheJavaException/HomeSystem/releases/latest";
					JSONParser parser = new JSONParser();
					URL github = new URL(url);
					URLConnection con = github.openConnection();
					HttpsURLConnection https = (HttpsURLConnection) con;
					
					if(https.getResponseCode() == 200) {
						BufferedReader in = new BufferedReader(new InputStreamReader(con.getInputStream()));
						
						String line;
						while((line = in.readLine()) != null) {
							JSONObject object = (JSONObject) parser.parse(line);
							String tag = object.get("tag_name").toString();

							if(!Data.version.equalsIgnoreCase(tag) && !tag.equalsIgnoreCase("beta")) {
								Log.write(Methods.createPrefix() + "Version " + tag + " is now available. Downloading...", true);
								
								String[] cmd = {"git", "clone",
												"https://github.com/TheJavaException/HomeSystem",
												"HomeSystem-" + tag};
								Process p = Runtime.getRuntime().exec(cmd);
								p.waitFor();
								Log.write("Finished downloading of Version " + tag, false);
							}
						}
						in.close();
					}
					Thread.sleep(wait * 60000);
				}
			}catch(IOException | ParseException | InterruptedException e) {
				e.printStackTrace();
				Log.write(createPrefix() + "Error in Methods(159): " + e.getMessage(), false);
			}
		}).start();
	}
	
}
