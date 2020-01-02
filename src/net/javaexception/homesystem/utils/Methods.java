package net.javaexception.homesystem.utils;

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
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

import javax.imageio.ImageIO;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import net.javaexception.homesystem.main.Main;
import net.sf.image4j.codec.ico.ICOEncoder;

public class Methods {
	
	public static String createPrefix() {
		if(Data.isServerStarted) {
			System.out.println("");
		}
		SimpleDateFormat date = new SimpleDateFormat("HH:mm:ss");
		String prefix = "[" + date.format(new Date()) + "] ";
		return prefix;
	}
	
	public static String getDate() {
		SimpleDateFormat date = new SimpleDateFormat("Y-MM-dd");
		return date.format(new Date());
	}
	
	public static int getTimeInSeconds() {
		Date now = new Date();
		Calendar calendar = Calendar.getInstance();
		
		calendar.setTime(now);
		
		int hourInSec = (calendar.get(Calendar.HOUR_OF_DAY) * 3600);
		int minuteInSec = (calendar.get(Calendar.MINUTE) * 60);
		int seconds = calendar.get(Calendar.SECOND);
		
		return (hourInSec + minuteInSec + seconds);
	}
	
	public static int getDateAsInt() {
		Date date = new Date();
		
		String stringDate= new SimpleDateFormat("yyyyMMdd").format(date);
		
		return Integer.parseInt(stringDate);
	}
	
	public static void extractWebsite() {
		Log.write(createPrefix() + "Extracting Website-Data...", true);
		
		File path = new File("HTTP");
		File faviconICO = new File(path + "//favicon.ico");
		File faviconPNG = new File(path + "//favicon.png");
		File index = new File(path + "//index.php");
		File home = new File(path + "//home.php");
		File style = new File(path + "//style.css");
		
		if(!path.exists()) {
			path.mkdir();
		}
		
		if(faviconICO.exists()) {
			faviconICO.delete();
		}
		
		if(faviconPNG.exists()) {
			faviconPNG.delete();
		}
		
		if(index.exists()) {
			index.delete();
		}
		
		if(home.exists()) {
			home.delete();
		}
		
		if(style.exists()) {
			style.delete();
		}
		
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
				if(format.equals("ico")) {
					BufferedImage bImage = ImageIO.read(Main.class.getResourceAsStream(resource));
					
					ICOEncoder.write(bImage, file);
				}else {
					BufferedImage bImage = ImageIO.read(Main.class.getResourceAsStream(resource));
					
					ImageIO.write(bImage, format, file);
				}
			}else {
				InputStream stream = Main.class.getResourceAsStream(resource);
				BufferedReader in = new BufferedReader(new InputStreamReader(stream, "UTF-8"));
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
			Log.write(createPrefix() + "Error in Methods(133): " + e.getMessage(), true);
			Log.write("", false);
		}
	}
	
	public static void startVersionChecking() {
		new Thread(() -> {
			try {
				while(true) {
					String url = "https://api.github.com/repos/TheJavaException/HomeSystem/releases/latest";
					JSONParser parser = new JSONParser();
					URL github = new URL(url);
					URLConnection con = github.openConnection();
					BufferedReader in = new BufferedReader(new InputStreamReader(con.getInputStream()));
					
					String line;
					while((line = in.readLine()) != null) {
						JSONObject object = (JSONObject) parser.parse(line);
						String tag = object.get("tag_name").toString();

						if(!Data.version.equalsIgnoreCase(tag) && !Data.newVersion) {
							Data.newVersion = true;
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
					
					Thread.sleep(5000);
				}
			}catch(IOException | ParseException | InterruptedException e) {
				e.printStackTrace();
				Log.write(createPrefix() + "Error in Methods(170): " + e.getMessage(), false);
			}
		}).start();
	}
	
}
