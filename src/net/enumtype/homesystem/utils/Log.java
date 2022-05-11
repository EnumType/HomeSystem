package net.enumtype.homesystem.utils;

import net.enumtype.homesystem.main.Main;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;

public class Log {
	
	private File latest;
	private File log;
	private File dataFolder;

	public Log() {
		try {
			init();
		}catch (IOException e) {
			if(Main.getData().printStackTraces()) e.printStackTrace();
			writeError(e);
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

	public void write(String output, boolean isNextLine, boolean prefix) {
		if(prefix) output = "[" + new SimpleDateFormat("HH:mm:ss").format(new Date()) + "] " + output;

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
			}catch (IOException e) {
				if(Main.getData().printStackTraces()) e.printStackTrace();
				writeError(e);
			}
		}else {
			System.out.println(createPrefix() + "Can't write to Logfile!");
			System.out.println(output);
		}
	}

	public void writeError(Exception e) {
		final String className = e.getStackTrace()[0].getClassName();
		int line = e.getStackTrace()[0].getLineNumber();

		write("Error in " + className + "(" + line + "): " + e.getMessage(), false, true);
	}

	public static void writeTestError(Exception e) {
		final String className = e.getStackTrace()[0].getClassName();
		int line = e.getStackTrace()[0].getLineNumber();

		System.out.println("Error in " + className + "(" + line + "): " + e.getMessage());
	}

	public String createPrefix() {
		return "[" + new SimpleDateFormat("HH:mm:ss").format(new Date()) + "] ";
	}
	
}
