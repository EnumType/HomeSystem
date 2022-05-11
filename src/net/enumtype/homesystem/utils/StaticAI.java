package net.enumtype.homesystem.utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.ZonedDateTime;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import net.enumtype.homesystem.main.Main;
import net.enumtype.homesystem.xmlrpc.Rooms;

public class StaticAI {
	
	public static int predict(String device, int brightness, long time) throws IOException {
		int state = 0;
		if(model.exists()) {
			String[] cmd = {"python3", model.getAbsolutePath(),
							device,
							"false", "true",
							"[" + brightness + "," + time + "]"};
			Process p = Runtime.getRuntime().exec(cmd);
			try {
				p.waitFor();
				
				int len;
				if ((len = p.getErrorStream().available()) > 0) {
				  byte[] buf = new byte[len];
				  final int i = p.getErrorStream().read(buf);
				  log.write("Command error:\t\""+new String(buf)+"\"; i=" + i, false);
				}
				
				state = p.exitValue();
			}catch(InterruptedException e) {
				p.destroy();
				e.printStackTrace();
				log.write(Methods.createPrefix() + "Error in AI(264): " + e.getMessage(), false);
			}
		}
		
		return state;
	}
	
	public static void train(String device) {
		new Thread(() -> {
			try {
				File data = new File("AI//data//" + device + ".csv");
				if(data.exists()) {
					String[] cmd = {"screen", "-dmS", "AI-" + device,
									"python3", "AI/Home-System.py",
									data.getName().replace(".csv", ""),
									"true",
									"false"};
					Runtime.getRuntime().exec(cmd);
				}
			}catch(IOException e) {
				e.printStackTrace();
				log.write(Methods.createPrefix() + "Error in AI(285): " + e.getMessage(), false);
			}
		}).start();
	}
	
}
