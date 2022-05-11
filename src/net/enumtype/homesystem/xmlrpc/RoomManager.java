package net.enumtype.homesystem.xmlrpc;

import net.enumtype.homesystem.main.Main;
import net.enumtype.homesystem.utils.Log;
import net.enumtype.homesystem.utils.Methods;
import org.yaml.snakeyaml.DumperOptions;
import org.yaml.snakeyaml.Yaml;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class RoomManager {

    private List<Room> rooms = new ArrayList();
    private final Log log;

    public RoomManager() {
        this.log = Main.getLog();
        load();
    }

    public void load() {
        try {
            //File file = new File("Rooms.yml");
            File file = new File("C:\\Users\\Anfinn\\Desktop\\Coding\\Java\\HomeSystem\\resources\\Rooms.yml");
            Map<Object, Map<Object, Map<Object, Map<Object, Object>>>> data;

            if(rooms != null) rooms.clear();
            if(!file.exists()) {
                log.write(Methods.createPrefix() + "Creating Rooms.yml...", true);
                InputStream resource = Main.class.getResourceAsStream("/Rooms.yml");
                Yaml in = new Yaml();
                data = (Map<Object, Map<Object, Map<Object, Map<Object, Object>>>>) in.load(resource);

                DumperOptions options = new DumperOptions();
                options.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK);
                options.setPrettyFlow(true);

                Yaml out = new Yaml(options);
                FileWriter writer = new FileWriter(file);
                out.dump(data, writer);
            }

            log.write(Methods.createPrefix() + "Loading Rooms.yml...", true);
            FileInputStream in = new FileInputStream(file);
            Yaml yaml = new Yaml();
            data = (Map<Object, Map<Object, Map<Object, Map<Object, Object>>>>) yaml.load(in);
            loadRooms(data);
        }catch(IOException e) {
            e.printStackTrace();
            log.write(Methods.createPrefix() + "Error in RoomManager(53): " + e.getMessage(), false);
        }
    }

    private void loadRooms(Map<Object, Map<Object, Map<Object, Map<Object, Object>>>> data) {
        for(Object name : data.keySet()) {
            Room r = new Room(name.toString(), ((Object) data.get(name).get("Permission")).toString(), data.get(name).get("Devices"));
        }
    }

    public boolean existsRoom(String name) {
        return getRoom(name) != null;
    }

    public Room getRoom(String name) {
        for(Room room : rooms) {
            if(room.getName().equalsIgnoreCase(name)) return room;
        }

        return null;
    }

    public List<Room> getRooms() {
        return rooms;
    }

}
