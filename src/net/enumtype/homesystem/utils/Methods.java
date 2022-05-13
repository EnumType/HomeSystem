package net.enumtype.homesystem.utils;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.math.BigInteger;
import java.net.URL;
import java.net.URLConnection;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import javax.imageio.ImageIO;
import javax.net.ssl.HttpsURLConnection;

import net.enumtype.homesystem.Main;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import net.sf.image4j.codec.ico.ICOEncoder;

public class Methods {
	
	public static String getDate() {
		return new SimpleDateFormat("yyyy-MM-dd").format(new Date());
	}
	
	public static long getUnixTime() {
		return System.currentTimeMillis() / 1000;
	}
	
	public static void extractWebsite() throws IOException {
		Main.getLog().write("Extracting Website-Data...");
		
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

		Main.getLog().write("Extracted Website-Data");
		System.out.print(Main.getCommandPrefix());
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
			Main.getLog().writeError(e);
		}
	}
	
	public static void startVersionChecking() {
		Executors.newSingleThreadScheduledExecutor().scheduleAtFixedRate(() -> {
			try {
				String url = "https://api.github.com/repos/EnumType/HomeSystem/releases/latest";
				JSONParser parser = new JSONParser();
				URL github = new URL(url);
				URLConnection con = github.openConnection();
				HttpsURLConnection https = (HttpsURLConnection) con;

				if (https.getResponseCode() == 200) {
					BufferedReader in = new BufferedReader(new InputStreamReader(con.getInputStream()));

					String line;
					while ((line = in.readLine()) != null) {
						JSONObject object = (JSONObject) parser.parse(line);
						String tag = object.get("tag_name").toString();

						if (!Main.getData().getVersion().equalsIgnoreCase(tag) && !tag.equalsIgnoreCase("beta")) {
							if(new File("HomeSystem-" + tag).exists()) continue;

							Main.getLog().write("Version " + tag + " is now available. Downloading...");

							String[] cmd = {"git", "clone", "https://github.com/EnumType/HomeSystem", "HomeSystem-" + tag};
							Process p = Runtime.getRuntime().exec(cmd);
							p.waitFor();
							Main.getLog().write("Finished downloading of Version " + tag);
							System.out.print(Main.getCommandPrefix());
						}
					}
					in.close();
				}
			}catch(IOException | ParseException | InterruptedException e) {
				Main.getLog().writeError(e);
			}
		}, 0, 5, TimeUnit.MINUTES);
	}

	public static String sha512(String input, String salt) {
		input = salt + input;
		try {
			final MessageDigest md = MessageDigest.getInstance("SHA-512");
			final byte[] messageDigest = md.digest(input.getBytes());
			return new BigInteger(1, messageDigest).toString(16);
		}catch(Exception e) {
			Main.getLog().writeError(e);
		}
		return "";
	}

	public static String sha512(String input) {
		return sha512(input, Main.getData().getHashSalt());
	}
	
}
