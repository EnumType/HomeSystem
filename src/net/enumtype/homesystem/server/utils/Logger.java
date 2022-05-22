package net.enumtype.homesystem.server.utils;

import net.enumtype.homesystem.server.HomeSystem;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.Date;

public class Logger extends PrintStream {

    private final PrintStream out;
    private String currentLine;
    private File latest;
    private File log;
    private File dataFolder;

    public Logger(PrintStream out) {
        super(out);
        this.out = out;
        init();
        System.setOut(this);
        System.setErr(new ErrorLogger(this));
    }

    public void init() {
        try {
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
        }catch(IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void print(String s) {
        clearLine();
        currentLine = s;
        out.print(s);
    }

    @Override
    public void println(String s) {
        clearLine();
        s = createPrefix() + s;
        out.println(s);
        saveToFile(s);
        print(HomeSystem.getCommandPrefix());
    }

    public void clearLine() {
        if(currentLine == null || currentLine.isEmpty()) return;

        for(int i = 0; i < currentLine.length(); i++) {
            out.print("\b");
        }

        currentLine = "";
    }

    public void saveToFile(String line) {
        if(latest == null || log == null || dataFolder == null) return;

        try {
            PrintWriter latestOut = new PrintWriter(new FileWriter(latest, true), true);
            PrintWriter logout = new PrintWriter(new FileWriter(log, true), true);
            latestOut.write(line + "\r\n");
            logout.write(line + "\r\n");

            latestOut.close();
            logout.close();
        }catch (IOException e) {
            e.printStackTrace();
        }
    }

    public String createPrefix() {
        return "[" + new SimpleDateFormat("HH:mm:ss").format(new Date()) + "] ";
    }
}

class ErrorLogger extends PrintStream {

    private final Logger logger;

    public ErrorLogger(Logger logger) {
        super(System.out);
        this.logger = logger;
    }

    @Override
    public void print(String s) {
        s = logger.createPrefix() + s;
        logger.saveToFile(s);
        super.print(s);
    }
}