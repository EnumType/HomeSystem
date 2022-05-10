package net.enumtype.homesystem.utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

public class Log {
	
	private static File latest;
	private static File log;
	private static File datafolder;
	
	public static void initLog() throws IOException{
		datafolder = new File("logs");
		if(!datafolder.exists()) if(!datafolder.mkdir()) throw new IOException("Cannot create directory!");
		
		latest = new File(datafolder + "//latest.log");

		if(latest.exists()) if(!latest.delete()) throw new IOException("Cannot delete file!");
		
		log = new File(datafolder + "//" + Methods.getDate() + ".log");
		if(log.exists()) {
			for(int i = 0; log.exists(); i++) {
				log = new File(datafolder + "//" + Methods.getDate() + "." + i + ".log");
			}
		}
	}
	
	public static void write(String output, boolean isNextLine) {
		if(latest != null && log != null && datafolder != null) {
			try {
				PrintWriter latestout = new PrintWriter(new FileWriter(latest, true), true);
				PrintWriter logout = new PrintWriter(new FileWriter(log, true), true);
				latestout.write(output + "\r\n");
				logout.write(output + "\r\n");
				System.out.println(output);
				
				if(!isNextLine) System.out.print(">");
				
				latestout.close();
				logout.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}else {
			System.out.println(Methods.createPrefix() + "Can't write to Logfile!");
			System.out.println(output);
		}
	}
	
}
