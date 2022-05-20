package net.enumtype.homesystem.rooms;

import net.enumtype.homesystem.HomeSystem;
import net.enumtype.homesystem.utils.AIException;
import net.enumtype.homesystem.utils.Methods;

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
    private final File aiDir;
    private final File dataDir;
    private final File modelTemplate;
    private final File modelPath;
    private final Map<Device, List<String>> deviceData; //Syntax -> Time,Light,Temperature,Special,State

    public AIManager() {
        this.aiDir = new File("AI");
        this.dataDir = new File(aiDir + "//data");
        this.modelTemplate = new File(aiDir + "//Home-System.py");
        this.modelPath = new File(aiDir + "//models");
        this.deviceData = new HashMap<>();
        init();
    }

    public void init() {
        try {
            if(!aiDir.exists()) if(!aiDir.mkdir()) throw new IOException("Cannot create directory!");
            if(!dataDir.exists()) if(!dataDir.mkdir()) throw new IOException("Cannot create directory!");
            if(!modelPath.exists()) if(!modelPath.mkdir()) throw new IOException("Cannot create directory!");

            if(modelTemplate.exists() && !modelTemplate.delete()) throw new IOException("Cannot renew file!");

            InputStream stream = HomeSystem.class.getResourceAsStream("/Home-System.py");
            BufferedReader in = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8));
            BufferedWriter out = new BufferedWriter(new FileWriter(modelTemplate));

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
        final RoomManager roomManager = HomeSystem.getRoomManager();
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

        predictionTimer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                for(Room room : HomeSystem.getRoomManager().getRooms()) {
                    for(Device device : room.getDevices()) {
                        try {
                            if(!device.aiControlled()) continue;

                            long time = Methods.getUnixTime();
                            double light = 0;
                            double temperature = 0;
                            double special = 0;
                            float prediction = device.getAI().predict(time, light, temperature, special);

                            if(Math.round(prediction) != Math.round(device.getState())) device.setValue(prediction);
                            //TODO: after testing remove Math.round()
                        }catch(AIException e) {
                            e.printStackTrace();
                        }
                    }
                }
            }
        }, 0, waitDelay * 60000L);
    }

    public void startAutoTraining() {
        final ZonedDateTime now = ZonedDateTime.now();
        if(trainingScheduler == null) trainingScheduler = Executors.newScheduledThreadPool(1);
        ZonedDateTime nextRun = now.withHour(0).withMinute(0).withSecond(0);

        if(now.compareTo(nextRun) > 0) nextRun = nextRun.plusDays(1);

        final Duration duration = Duration.between(now, nextRun);
        long initialDelay = duration.getSeconds();

        trainingScheduler.scheduleAtFixedRate(this::trainAll,
                initialDelay, TimeUnit.DAYS.toSeconds(1), TimeUnit.SECONDS);
    }

    public void stopDataSaving() {
        savingTimer.cancel();
        savingTimer = null;
    }

    public void stopPredictions() {
        predictionTimer.cancel();
        predictionTimer = null;
    }

    public void stopAutoTraining() {
        trainingScheduler.shutdown();
        trainingScheduler = null;
    }

    public void interruptAll() {
        for(Room room : HomeSystem.getRoomManager().getRooms()) {
            for(Device device : room.getDevices()) if(device.collectData()) device.getAI().interrupt();
        }
    }

    public void trainAll() {
        saveData();
        System.out.println("Starting AI training");
        for(Room room : HomeSystem.getRoomManager().getRooms()) {
            for(Device device : room.getDevices()) if(device.collectData()) device.getAI().train();
        }
    }

    public void saveData() {
        System.out.println("Saving AI training data...");
        try {
            final RoomManager roomManager = HomeSystem.getRoomManager();

            for(Room room : roomManager.getRooms()) {
                for(Device device : room.getDevices()) {
                    if(!deviceData.containsKey(device)) continue;

                    final File data = new File(dataDir + "//" + room.getName().replaceAll(" ", "-") +
                            "-" + device.getName().replaceAll(" ", "-") + ".csv");
                    final FileWriter writer = new FileWriter(data, true);
                    final PrintWriter out = new PrintWriter(writer);

                    if(!data.exists()) out.println("Time,Light,Temperature,Special,State");
                    for(String line : deviceData.get(device)) out.println(line);
                    writer.flush();
                    writer.close();
                    out.close();
                }
            }
            System.out.println("Finished saving");
        }catch(IOException e) {
            e.printStackTrace();
        }
    }

    public File getModelTemplate() {return modelTemplate;}

}
