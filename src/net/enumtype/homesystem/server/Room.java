package net.enumtype.homesystem.server;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Room {

    private final String name;
    private final String permission;
    private final List<Device> devices = new ArrayList<>();

    public Room(Object name, Object permission, Map<Object, Map<Object, Object>> deviceData) {
        this.name = name.toString().replaceAll(" ", "-");
        this.permission = permission.toString();

        for(Object deviceName : deviceData.keySet()) {
            devices.add(new Device(deviceName.toString(), this.name, deviceData.get(deviceName)));
        }
    }

    public boolean hasDevice(String name) {
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

    public List<String> getDeviceNames() {
        final List<String> names = new ArrayList<>();

        for(Device device : devices) {
            names.add(device.getName());
        }

        return names;
    }

    public Device getDevice(String name) {
        for(Device device : devices) {
            if(device.getName().equalsIgnoreCase(name)) return device;
        }

        return null;
    }

}