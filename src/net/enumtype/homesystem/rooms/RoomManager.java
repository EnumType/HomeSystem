package net.enumtype.homesystem.rooms;

import net.enumtype.homesystem.Main;
import net.enumtype.homesystem.utils.Log;
import org.yaml.snakeyaml.DumperOptions;
import org.yaml.snakeyaml.Yaml;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class RoomManager {

    private final List<Room> rooms = new ArrayList<>();
    private final Log log;

    public RoomManager() {
        this.log = Main.getLog();
        load();
    }

    @SuppressWarnings("unchecked")
    public void load() {
        try {
            File file = new File("Rooms.yml");
            Map<Object, Map<Object, Map<Object, Map<Object, Object>>>> data;

            rooms.clear();
            if(!file.exists()) {
                log.write("Creating Rooms.yml...", true, true);
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

            log.write("Loading Rooms.yml...", true, true);
            FileInputStream in = new FileInputStream(file);
            Yaml yaml = new Yaml();
            data = (Map<Object, Map<Object, Map<Object, Map<Object, Object>>>>) yaml.load(in);
            loadRooms(data);
        }catch(IOException e) {
            log.writeError(e);
        }
    }

    private void loadRooms(Map<Object, Map<Object, Map<Object, Map<Object, Object>>>> data) {
        for(Object name : data.keySet()) {
            System.out.println(data.get(name).get("Permission").toString());
            Room room = new Room(name.toString(),
                    data.get(name).get("Permission").entrySet().iterator().next().getKey().toString(),
                    data.get(name).get("Devices"));
            rooms.add(room);
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

    public Room getRoomByDevice(Device device) {
        for(Room room : getRooms()) {
            if(room.getDevices().contains(device)) return room;
        }

        return null;
    }

    public List<Room> getRooms() {
        return rooms;
    }

}
