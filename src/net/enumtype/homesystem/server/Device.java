package net.enumtype.homesystem.server;

import net.enumtype.homesystem.server.utils.Data;
import org.apache.xmlrpc.XmlRpcException;
import org.apache.xmlrpc.client.XmlRpcClient;
import org.apache.xmlrpc.client.XmlRpcClientConfigImpl;

import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Device {

    private final String name;
    private final boolean collectData;
    private final boolean aiControl;
    private final DeviceType type;
    private final AI ai;
    private String address;
    private boolean hmIp;

    public Device() {
        this.name = "";
        this.collectData = false;
        this.aiControl = false;
        this.type = DeviceType.BRIGHT;
        this.ai = null;
    }

    public Device(String name, String roomName, Map<Object, Object> optionData) {
        this.name = name.replaceAll(" ", "-");
        this.address = optionData.get("Address").toString();
        this.hmIp = Boolean.parseBoolean(optionData.get("HmIP").toString());
        this.collectData = Boolean.parseBoolean(optionData.get("AIData").toString());
        this.aiControl = Boolean.parseBoolean(optionData.get("AIControl").toString());
        this.type = DeviceType.valueOf(optionData.get("Type"));

        this.ai = new AI(roomName, name);
    }

    public void setAddress(String address) {
        this.address = address;
    }

    public void setHmIp(boolean hmIp) {
        this.hmIp = hmIp;
    }

    public void stop() {
        setValue("STOP", true);
    }

    public void setValue(Object value_key, Object value) {
        new Thread(() -> {
            try {
                final Data data = HomeSystem.getData();
                XmlRpcClientConfigImpl config = new XmlRpcClientConfigImpl();
                XmlRpcClient xmlRpcClient = new XmlRpcClient();

                config.setServerURL(new URL("http://" + data.getXmlRpcAddress()+ ":" +
                        (hmIp ? data.getHmIpPort() : data.getXmlRpcPort())));
                xmlRpcClient.setConfig(config);

                if(getValue(type.getValueKey()) != value)
                    xmlRpcClient.execute("setValue", new Object[]{address, value_key, value});
            }catch(MalformedURLException | XmlRpcException e) {
                e.printStackTrace();
            }
        }).start();
    }

    public void setValue(Object value) {
        setValue(type.getValueKey(), value);
    }

    public String getValue(String value_key) {
        try {
            final Data data = HomeSystem.getData();
            XmlRpcClientConfigImpl config = new XmlRpcClientConfigImpl();
            XmlRpcClient client = new XmlRpcClient();

            config.setServerURL(new URL("http://" + data.getXmlRpcAddress()+ ":" +
                    (hmIp ? data.getHmIpPort() : data.getXmlRpcPort())));
            client.setConfig(config);

            return client.execute("getValue", new Object[]{address, value_key}).toString();
        }catch(MalformedURLException | XmlRpcException e) {
            e.printStackTrace();
        }

        return "";
    }

    public double getState() {
        final String result = getValue(type.getValueKey());
        switch (type) {
            case LAMP:
                return Boolean.parseBoolean(result) ? 1 : 0;
            case ROLL:
                return Math.round(Float.parseFloat(result));
            default:
                return 0;
        }
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

    public String getName() {return name;}
    public AI getAI() {return ai;}
    public DeviceType getType() {return type;}
    public boolean aiControlled() {return aiControl;}
    public boolean collectData() {return collectData;}

}

enum DeviceType {
    LAMP("STATE"), ROLL("LEVEL"), BRIGHT("STATE")/*TODO: Check value_key*/, UNKNOWN("");

    private final String value_key;
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

    public String getValueKey() {return value_key;}
}