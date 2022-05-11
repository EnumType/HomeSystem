package net.enumtype.homesystem.utils;

import net.enumtype.homesystem.main.Main;
import net.enumtype.homesystem.xmlrpc.Device;
import net.enumtype.homesystem.xmlrpc.Room;
import net.enumtype.homesystem.xmlrpc.RoomManager;
import net.enumtype.homesystem.xmlrpc.Rooms;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.ZonedDateTime;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class AIManager {

    private Timer savingTimer;
    private Timer predictionTimer;
    private ScheduledExecutorService trainingScheduler;
    private final Log log;
    private final Data data;
    private final File aiDir;
    private final File dataDir;
    private final File model;
    private final File modelPath;
    private final Map<Device, List<String>> deviceData; //Syntax -> Time,Light,Temperature,Special,State

    public AIManager() {
        this.log = Main.getLog();
        this.data = Main.getData();
        this.aiDir = new File("AI");
        this.dataDir = new File(aiDir + "//data");
        this.model = new File(aiDir + "//Home-System.py");
        this.modelPath = new File(aiDir + "//models");
        this.deviceData = new HashMap<>();
        init();
    }

    public void init() {
        try {
            if(!aiDir.exists()) if(!aiDir.mkdir()) throw new IOException("Cannot create directory!");
            if(!dataDir.exists()) if(!dataDir.mkdir()) throw new IOException("Cannot create directory!");
            if(!modelPath.exists()) if(!modelPath.mkdir()) throw new IOException("Cannot create directory!");

            if(model.exists() && !model.delete()) throw new IOException("Cannot renew file!");

            InputStream stream = Main.class.getResourceAsStream("/Home-System.py");
            BufferedReader in = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8));
            BufferedWriter out = new BufferedWriter(new FileWriter(model));

            String line;
            while((line = in.readLine()) != null) out.write(line + "\r\n");

            in.close();
            out.close();
        }catch(IOException e) {
            e.printStackTrace();
        }
    }

    /**
     *
     * @param waitDelay Sleeping delay in minutes
     */
    public void startDataSaving(int waitDelay) {
        if(savingTimer == null) savingTimer = new Timer();
        final RoomManager roomManager = Main.getRoomManager();
        final StringBuilder builder = new StringBuilder();

        savingTimer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                for(Room room : roomManager.getRooms()) {
                    for(Device device : room.getDevices()) {
                        if(!device.collectData()) continue;

                        builder.append(Methods.getUnixTime()); //Time
                        builder.append(",");
                        builder.append(0); //TODO: Light
                        builder.append(",");
                        builder.append(0); //TODO: Temperature
                        builder.append(",");
                        builder.append(0); //TODO: Special
                        builder.append(",");
                        builder.append(device.getState()); //State

                        if(!deviceData.containsKey(device)) deviceData.put(device, new ArrayList<>());
                        deviceData.get(device).add(builder.toString());
                        builder.setLength(0);
                    }
                }
            }
        }, 0, waitDelay * 60000L);
    }

    /**
     *
     * @param waitDelay Sleeping delay in minutes
     */
    public void startPredictions(int waitDelay) {
        if(predictionTimer == null) predictionTimer = new Timer();
        final RoomManager roomManager = Main.getRoomManager();

        predictionTimer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                for(Device device : deviceData.keySet()) {
                    if(!device.aiControlled()) continue;

                    long time = Methods.getUnixTime();
                    double light = 0;
                    double temperature = 0;
                    double special = 0;
                    float prediction = device.getAI().predict(time, light, temperature, special);

                    if(Math.round(prediction) != Math.round(device.getState())) device.setValue(prediction);
                    //TODO: after testing remove Math.round()
                }
            }
        }, 0, waitDelay * 60000L);
    }

    public void startAutoTraining() {
        final ZonedDateTime now = ZonedDateTime.now();
        final RoomManager roomManager = Main.getRoomManager();
        trainingScheduler = Executors.newScheduledThreadPool(1);
        ZonedDateTime nextRun = now.withHour(0).withMinute(0).withSecond(0);

        if(now.compareTo(nextRun) > 0) nextRun = nextRun.plusDays(1);

        final Duration duration = Duration.between(now, nextRun);
        long initialDelay = duration.getSeconds();

        trainingScheduler.scheduleAtFixedRate(() -> {
            log.write(Methods.createPrefix() + "Starting AI training", false);
            for(Device device : deviceData.keySet()) device.getAI().train();
        }, initialDelay, TimeUnit.DAYS.toSeconds(1), TimeUnit.SECONDS);
    }

    public void trainAll() {
        final RoomManager roomManager = Main.getRoomManager();
        saveData();

        for(Room room : roomManager.getRooms()) {
            for(Device device : room.getDevices()) {
                if(!device.collectData()) continue;

                device.getAI().train();
            }
        }
    }

    public void saveData() {
        try {
            final RoomManager roomManager = Main.getRoomManager();

            for(Room room : roomManager.getRooms()) {
                for(Device device : room.getDevices()) {
                    if(deviceData.containsKey(device)) {
                        final File data = new File(dataDir + "//" + room.getName() + "-" +
                                device.getName().replaceAll(" ", "-") + ".csv");
                        final FileWriter writer = new FileWriter(data, true);
                        final PrintWriter out = new PrintWriter(writer);

                        if(!data.exists()) out.println("Time,Light,Temperature,Special,State");
                        for(String line : deviceData.get(device)) out.println(line);
                        writer.flush();
                        writer.close();
                        out.close();
                    }
                }
            }
        }catch(IOException e) {
            e.printStackTrace();
        }
    }

    public void stopDataSaving() {
        savingTimer.cancel();
    }

    public void stopPredictions() {
        predictionTimer.cancel();
    }

    public void stopAutoTraining() {
        trainingScheduler.shutdownNow();
    }

    public List<AI> getAll() {
        final List<AI> ais = new ArrayList<>();

        for(Room room : Main.getRoomManager().getRooms()) {
            for(Device device : room.getDevices()) {
                ais.add(device.getAI());
            }
        }

        return ais;
    }

}
