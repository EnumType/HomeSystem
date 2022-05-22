package net.enumtype.homesystem.server.utils;

import java.io.*;
import java.math.BigInteger;
import java.net.URL;
import java.net.URLConnection;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import javax.net.ssl.HttpsURLConnection;

import net.enumtype.homesystem.server.HomeSystem;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

public class Methods {
	
	public static String getDate() {
		return new SimpleDateFormat("yyyy-MM-dd").format(new Date());
	}
	
	public static long getUnixTime() {
		return System.currentTimeMillis() / 1000;
	}
	
	public static void extractWebsite() throws IOException {
		System.out.println("Extracting Website-Data...");
		
		final File path = new File("HTTP");
		final File faviconICO = new File(path + "//favicon.ico");
		final File faviconPNG = new File(path + "//favicon.png");
		final File index = new File(path + "//index.html");
		final File home = new File(path + "//home.html");
		final File style = new File(path + "//style.css");
		final File script = new File(path + "//script.js");
		
		if(!path.exists()) if(!path.mkdir()) throw new IOException("Cannot create directory!");
		if(faviconICO.exists()) if(!faviconICO.delete()) throw new IOException("Cannot delete file!");
		if(faviconPNG.exists()) if(!faviconPNG.delete()) throw new IOException("Cannot delete file!");
		if(index.exists()) if(!index.delete()) throw new IOException("Cannot delete file!");
		if(home.exists()) if(!home.delete()) throw new IOException("Cannot delete file!");
		if(style.exists()) if(!style.delete()) throw new IOException("Cannot delete file!");
		if(script.exists()) if(!script.delete()) throw new IOException("Cannot delete file!");
		
		writeResources(faviconICO, "/HTML/" + faviconICO.getName(), true, "ico");
		writeResources(faviconPNG, "/HTML/" + faviconPNG.getName(), true, "png");
		writeResources(index, "/HTML/" + index.getName(), false, "html");
		writeResources(home, "/HTML/" + home.getName(), false, "html");
		writeResources(style, "/HTML/" + style.getName(), false, "css");
		writeResources(script, "/HTML/" + script.getName(), false, "css");

		System.out.println("Extracted Website-Data");
	}
	
	public static void writeResources(File file, String resource, boolean image, String format) {
		try {
			final InputStream stream = HomeSystem.class.getResourceAsStream(resource);
			if(stream == null) throw new IOException(resource + " not found!");

			if(image) {
				final byte[] data = stream.readAllBytes();
				final DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)));
				out.write(data);
				out.close();
			}else {
				BufferedReader in = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8));
				BufferedWriter out = new BufferedWriter(new FileWriter(file));
				
				String line;
				while((line = in.readLine()) != null) {
					out.write(line + "\r\n");
				}
				
				in.close();
				out.close();
			}

			stream.close();
		}catch(IOException e) {
			e.printStackTrace();
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

						if (!HomeSystem.getData().getVersion().equalsIgnoreCase(tag) && !tag.equalsIgnoreCase("beta")) {
							if(new File("HomeSystem-" + tag).exists()) continue;

							System.out.println("Version " + tag + " is now available. Downloading...");

							String[] cmd = {"git", "clone", "https://github.com/EnumType/HomeSystem", "HomeSystem-" + tag};
							Process p = Runtime.getRuntime().exec(cmd);
							p.waitFor();
							System.out.println("Finished downloading of Version " + tag);
						}
					}
					in.close();
				}
			}catch(IOException | ParseException | InterruptedException e) {
				e.printStackTrace();
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
			e.printStackTrace();
		}
		return "";
	}

	public static String sha512(String input) {
		return sha512(input, HomeSystem.getData().getHashSalt());
	}
	
}