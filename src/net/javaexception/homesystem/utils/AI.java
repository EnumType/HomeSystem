package net.javaexception.homesystem.utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import net.javaexception.homesystem.main.Main;

public class AI {
	
	public static void checkAIData() throws IOException {
		File AIDir = new File("AI");
		File DataDir = new File(AIDir + "//data");
		File model = new File(AIDir + "//Home-System.py");
		File modelPath = new File(AIDir + "//models");
		
		if(!AIDir.exists()) {
			AIDir.mkdir();
		}
		
		if(!DataDir.exists()) {
			DataDir.mkdir();
		}
		
		if(!modelPath.exists()) {
			modelPath.mkdir();
		}
		
		if(!model.exists()) {
			InputStream stream = Main.class.getResourceAsStream("/Home-System.py");
			BufferedReader in = new BufferedReader(new InputStreamReader(stream, "UTF-8"));
			BufferedWriter out = new BufferedWriter(new FileWriter(model));
			
			String line;
			while((line = in.readLine()) != null) {
				out.write(line + "\r\n");
			}
			
			in.close();
			out.close();
		}
	}
	
	public static int predict(String device, int brightness, int date, int time) throws IOException {
		int state = 0;
		File model = new File("AI//Home-System.py");
		if(model.exists()) {
			String cmd = "python3.7 " + model.getAbsolutePath() + " " +
							device + " " +
							"false " + "true " +
							"[" + brightness + "," + date + "," + time + "]";
			Process p = Runtime.getRuntime().exec(cmd);
			try {
				p.waitFor();
				
				int len;
				if ((len = p.getErrorStream().available()) > 0) {
				  byte[] buf = new byte[len];
				  p.getErrorStream().read(buf);
				  Log.write("Command error:\t\""+new String(buf)+"\"", false);
				}
				
				state = p.exitValue();
			}catch(InterruptedException e) {
				p.destroy();
				e.printStackTrace();
				Log.write(Methods.createPrefix() + "Error in AI(86): " + e.getMessage(), false);
			}
		}
		
		return state;
	}
	
	public static void train(String device) throws IOException {
		File model = new File("AI//Home-System.py");
		if(model.exists()) {
			String[] cmd = {"python3.7",
							model.getName(),
							"true",
							"false"};
			Runtime.getRuntime().exec(cmd);
		}
	}
	
}
