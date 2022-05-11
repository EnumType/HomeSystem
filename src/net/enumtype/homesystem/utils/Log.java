package net.enumtype.homesystem.utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

public class Log {
	
	private File latest;
	private File log;
	private File dataFolder;

	public Log() {
		try {
			init();
		}catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void init() throws IOException{
		dataFolder = new File("logs");
		if(!dataFolder.exists()) if(!dataFolder.mkdir()) throw new IOException("Cannot create directory!");
		
		latest = new File(dataFolder + "//latest.log");

		if(latest.exists()) if(!latest.delete()) throw new IOException("Cannot delete file!");
		
		log = new File(dataFolder + "//" + Methods.getDate() + ".log");
		if(log.exists()) {
			for(int i = 0; log.exists(); i++) {
				log = new File(dataFolder + "//" + Methods.getDate() + "." + i + ".log");
			}
		}
	}
	
	public void write(String output, boolean isNextLine) {
		if(latest != null && log != null && dataFolder != null) {
			try {
				PrintWriter latestOut = new PrintWriter(new FileWriter(latest, true), true);
				PrintWriter logout = new PrintWriter(new FileWriter(log, true), true);
				latestOut.write(output + "\r\n");
				logout.write(output + "\r\n");
				System.out.println(output);
				
				if(!isNextLine) System.out.print(">");

				latestOut.close();
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
