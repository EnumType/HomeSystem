package net.enumtype.homesystem.rooms;

import net.enumtype.homesystem.Main;
import net.enumtype.homesystem.utils.Log;

import java.io.File;
import java.io.IOException;

public class AI {

    private final String modelName;
    private final File modelTemplate;
    private final Log log;
    private Process modelProcess;

    public AI(String roomName, String deviceName) {
        this.modelName = roomName + deviceName;
        this.modelTemplate = Main.getAiManager().getModelTemplate();
        this.log = Main.getLog();
    }

    public void train() {
        new Thread(() -> {
            try {
                final File data = new File("AI//data//" + modelName + ".csv");
                if(!data.exists()) return;

                String[] cmd = {"screen", "-dmS", "AI-" + modelName,
                        "python3", "AI/Home-System.py",
                        data.getName().replace(".csv", ""),
                        "true",
                        "false"};
                Runtime.getRuntime().exec(cmd);
            }catch(IOException e) {
                log.writeError(e);
            }
        }).start();
    }

    public float predict(long time, double light, double temperature, double special) throws AIException{
        if(!modelTemplate.exists()) throw new AIException("No model template found!");

        try {
            String[] cmd = {"python3", modelTemplate.getAbsolutePath(),
                    modelName,
                    "false", "true",
                    "[" + time + "," + light + "," + temperature + "," + special +  "]"};
            modelProcess = Runtime.getRuntime().exec(cmd);
            modelProcess.waitFor();

            int len = modelProcess.getErrorStream().available();
            if (len > 0) {
                byte[] buf = new byte[len];
                final int i = modelProcess.getErrorStream().read(buf);
                log.write("Command error:\t\""+new String(buf)+"\"; i=" + i, false, true);
            }

            return modelProcess.exitValue() / 100F;
        }catch(InterruptedException | IOException e) {
            modelProcess.destroy();
            log.writeError(e);
        }

        return 0F;
    }

}