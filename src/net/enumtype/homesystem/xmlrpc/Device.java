package net.enumtype.homesystem.xmlrpc;

import net.enumtype.homesystem.utils.AI;
import net.enumtype.homesystem.server.Client;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Device {

    private final String name;
    private final String address;
    private final boolean hmip;
    private final boolean aidata;
    private final boolean aicontrol;
    private final DeviceType type;
    private final AI ai;

    public Device(String name, Map<Object, Object> optionData) {
        this.name = name;
        this.address = optionData.get("Address").toString();
        this.hmip = Boolean.parseBoolean(optionData.get("HmIP").toString());
        this.aidata = Boolean.parseBoolean(optionData.get("AIData").toString());
        this.aicontrol = Boolean.parseBoolean(optionData.get("AIControl").toString());
        this.type = DeviceType.valueOf(optionData.get("Type"));

        this.ai = new AI();
    }

    public void setValue(Object value, Client client) {
        XmlRpcServer.setValue(address, type.getValueKey(), value, client, hmip);
    }

    public void stop(Client client) {
        XmlRpcServer.setValue(address, "STOP", true, client, hmip);
    }

    public String getValue(String value_key) {
        return XmlRpcServer.getValue(address, value_key, hmip).toString();
    }

    public List<String> getStates() {
        final ArrayList<String> states = new ArrayList<>();

        if(type.equals(DeviceType.ROLL)) {
            states.add("LEVEL:" + getValue("LEVEL"));
            states.add("WORKING:" + getValue("WORKING"));
        }else if(type.equals(DeviceType.LAMP)) {
            states.add("STATE:" + getValue("STATE"));
        }

        return states;
    }

    public String getName() {
        return name;
    }

    public String getAddress() {
        return address;
    }

    public AI getAI() {
        return ai;
    }

    public DeviceType getType() {
        return type;
    }

}

enum DeviceType {
    LAMP("LEVEL"), ROLL("STATE"), UNKNOWN("");

    private String value_key;
    DeviceType(String value_key) {
        this.value_key = value_key;
    }

    public static DeviceType valueOf(Object object) {
        switch (object.toString().toLowerCase()) {
            case "lamp":
                return LAMP;
            case "roll":
                return ROLL;
            default:
                return UNKNOWN;
        }
    }

    public String getValueKey() {
        return value_key;
    }
}
