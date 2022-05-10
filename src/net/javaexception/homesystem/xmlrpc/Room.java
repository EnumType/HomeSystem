package net.javaexception.homesystem.xmlrpc;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Room {

    private String name;
    private String permission;
    private List<Device> devices = new ArrayList<>();

    public Room(String name, String permission, Map<Object, Map<Object, Object>> deviceData) {
        this.name = name;
        this.permission = permission;

        for(Object deviceName : deviceData.keySet()) {
            devices.add(new Device(deviceName.toString(), deviceData.get(deviceName)));
        }
    }

    public boolean existsDevice(String name) {
        return getDevice(name) != null;
    }

    public String getName() {
        return name;
    }

    public String getPermission() {
        return permission;
    }

    public List<Device> getDevices() {
        return devices;
    }

    public Device getDevice(String name) {
        for(Device device : devices) {
            if(device.getName().equalsIgnoreCase(name)) return device;
        }

        return null;
    }

}
