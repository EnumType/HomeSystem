package net.enumtype.homesystem.server;

import net.enumtype.homesystem.server.exceptions.AIException;

import java.io.File;
import java.io.IOException;

public class AI {

    private final String modelName;
    private final File modelTemplate;
    private Process predictionProcess;
    private Process trainingProcess;

    public AI(String roomName, String deviceName) {
        this.modelName = (roomName + deviceName).replaceAll(" ", "-");
        this.modelTemplate = HomeSystem.getAiManager().getModelTemplate();
    }

    public void train() {
        new Thread(() -> {
            try {
                final File data = new File("AI//data//" + modelName + ".csv");
                if(!data.exists()) return;

                String[] cmd = {//"screen", "-dmS", "AI-" + modelName, TODO: Check if its working without screen
                        "python3", "AI/Home-System.py",
                        data.getName().replace(".csv", ""),
                        "true",
                        "false"};
                trainingProcess = Runtime.getRuntime().exec(cmd);
                trainingProcess.waitFor();
                trainingProcess.destroy();
            }catch(InterruptedException | IOException e) {
                e.printStackTrace();
            }
        }).start();
    }

    public void interrupt() {
        if(trainingProcess != null && trainingProcess.isAlive()) trainingProcess.destroy();
        if(predictionProcess != null && predictionProcess.isAlive()) predictionProcess.destroy();
    }

    public float predict(long time, double light, double temperature, double special) throws AIException {
        if(!modelTemplate.exists()) throw new AIException("No model template found!");

        try {
            String[] cmd = {"python3", modelTemplate.getAbsolutePath(),
                    modelName,
                    "false", "true",
                    "[" + time + "," + light + "," + temperature + "," + special +  "]"};
            predictionProcess = Runtime.getRuntime().exec(cmd);
            predictionProcess.waitFor();

            int len = predictionProcess.getErrorStream().available();
            if (len > 0) {
                byte[] buf = new byte[len];
                final int i = predictionProcess.getErrorStream().read(buf);
                System.out.print("Command error:\t\""+new String(buf)+"\"; i=" + i);
            }

            predictionProcess.destroy();
            return predictionProcess.exitValue() / 100F;
        }catch(InterruptedException | IOException e) {
            predictionProcess.destroy();
            e.printStackTrace();
        }

        return 0F;
    }

}