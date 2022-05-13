package net.enumtype.homesystem.utils;

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

	public void write(String output) {
		write(output, true);
	}

	public void write(String output, boolean printToConsole) {
		output = createPrefix() + output;

		if(latest != null && log != null && dataFolder != null) {
			try {
				PrintWriter latestOut = new PrintWriter(new FileWriter(latest, true), true);
				PrintWriter logout = new PrintWriter(new FileWriter(log, true), true);
				latestOut.write(output + "\r\n");
				logout.write(output + "\r\n");
				if (printToConsole) System.out.println(output);

				latestOut.close();
				logout.close();
			}catch (IOException e) {
				writeError(e);
			}
		}else {
			System.out.println(createPrefix() + "Can't write to Logfile!");
			System.out.println(output);
		}
	}

	public void writeError(Exception e) {
		final StackTraceElement[] trace = e.getStackTrace();
		write(e.getClass().getSimpleName() + ": " + e.getMessage());

		write(e.toString());
		for(StackTraceElement stackTraceElement : trace) write("    " + stackTraceElement.toString());
		System.out.print("> ");
	}

	public void writeError(Throwable t) {
		final StackTraceElement[] trace = t.getStackTrace();
		write(t.getClass().getSimpleName() + ": " + t.getMessage());

		write(t.toString());
		for(StackTraceElement stackTraceElement : trace) write("    " + stackTraceElement.toString());
		System.out.print("> ");
	}

	public String createPrefix() {
		return "[" + new SimpleDateFormat("HH:mm:ss").format(new Date()) + "] ";
	}
	
}
