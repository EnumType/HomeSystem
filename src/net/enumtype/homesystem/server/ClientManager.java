package net.enumtype.homesystem.server;

import net.enumtype.homesystem.websocket.WebSocket;
import net.enumtype.homesystem.main.Main;
import net.enumtype.homesystem.utils.Log;
import net.enumtype.homesystem.utils.Methods;
import org.eclipse.jetty.websocket.api.Session;
import org.yaml.snakeyaml.DumperOptions;
import org.yaml.snakeyaml.Yaml;

import java.io.*;
import java.net.InetAddress;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ClientManager {

    private final Map<String, String> userData = new HashMap<>();
    private final Map<String, List<String>> userPermissions = new HashMap<>();
    private final List<Client> clients = new ArrayList<>();

    public void load() throws IOException {
        loadUserData();
        loadUserPerm();
    }

    public void loadUserData() throws IOException {
        if(!userData.isEmpty()) userData.clear();

        File file = new File("User-Data");
        File dataFile = new File(file + "//data.txt");
        if(!file.exists()) if(!file.mkdirs()) throw new IOException("Cannot create directories");
        if(!dataFile.exists()) {
            registerUser("a", "a");
            removeUser("a", "a");
        }

        BufferedReader reader = new BufferedReader(new FileReader(file + "//data.txt"));
        boolean end = false;
        while(!end) {
            String user = reader.readLine();
            String pass = reader.readLine();

            if(user == null && pass == null) {
                end = true;
            }else if( user != null && user.startsWith("Name: ") && pass.startsWith("Password: ")){
                userData.put(user.replaceAll("Name: ", ""), pass.replaceAll("Password: ", ""));
            }
        }

        reader.close();
    }

    public void registerUser(String username, String password) throws IOException {
        PrintWriter out = new PrintWriter(new FileWriter("User-Data//data.txt", true), true);
        out.write("Name: " + username + "\r\n");
        out.write("Password: " + password + "\r\n");
        out.close();
    }

    public void removeUser(String username, String password) throws IOException {
        File data = new File("User-Data//data.txt");
        File newData = new File("User-Data//data2.txt");
        BufferedReader reader = new BufferedReader(new FileReader(data));
        BufferedWriter writer = new BufferedWriter(new FileWriter(newData));
        boolean found = false;
        String line;

        while((line = reader.readLine()) != null) {
            if(!found) {
                if(!line.equals("Name: " + username) && !line.equals("Password: " + password) && !line.equals("\r\n")) {
                    writer.write(line + "\r\n");
                }else {
                    found = true;
                    reader.readLine();
                }
            }else {
                writer.write(line + "\r\n");
            }
        }

        reader.close();
        writer.close();
        if(!data.delete()) throw new IOException("Cannot delete file!");
        if(!newData.renameTo(data)) throw new IOException("Cannot rename file!");
    }

    public void loadUserPerm() throws IOException {
        File file = new File("Permissions.yml");

        if(!userPermissions.isEmpty()) userPermissions.clear();
        if(!file.exists()) {
            Log.write(Methods.createPrefix() + "Creating Permissions.yml...", true);
            InputStream resource = Main.class.getResourceAsStream("/Permissions.yml");
            Yaml in = new Yaml();
            Map<String, List<String>> map = (Map<String, List<String>>) in.load(resource);

            DumperOptions options = new DumperOptions();
            options.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK);
            options.setPrettyFlow(true);

            Yaml out = new Yaml(options);
            FileWriter writer = new FileWriter(file);
            out.dump(map, writer);
        }

        Log.write(Methods.createPrefix() + "Loading Permissions.yml...", true);
        Yaml yaml = new Yaml();
        FileInputStream io = new FileInputStream(new File("Permissions.yml"));

        Map<Object, List<String>> list = (Map<Object, List<String>>) yaml.load(io);

        list.keySet().forEach(user -> userPermissions.put(user.toString(), list.get(user)));
    }

    public void writeUserPerm() throws IOException {
        Log.write(Methods.createPrefix() + "Saving Permissions.yml...", true);
        DumperOptions options = new DumperOptions();
        options.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK);
        options.setPrettyFlow(true);

        Yaml yaml = new Yaml(options);
        FileWriter writer = new FileWriter("Permissions.yml");
        yaml.dump(userPermissions, writer);
    }

    public boolean verifyLoginData(InetAddress address, String username, String password) {
        if(userData.containsKey(username) && userData.get(username).equals(password)) {
            Log.write(Methods.createPrefix() + "User '" + username + "' logged in with IP " + address.toString(), false);
            return true;
        }else {
            Log.write(Methods.createPrefix() + "User tried to login:", true);
            Log.write(Methods.createPrefix() + "USERNAME: " + username, true);
            Log.write(Methods.createPrefix() + "IPADDRESS: " + address.toString(), true);
            Log.write("", false);
            return false;
        }
    }

    public void loginClient(Session session, String username, String password) {
        final Client client = new Client(session, username, userPermissions.containsKey(username) ? userPermissions.get(username) : new ArrayList<>());
        if(verifyLoginData(session.getRemoteAddress().getAddress(), username, password)) {
            clients.add(client);
            client.sendMessage("verifylogin " + client.getName());
        }else client.sendMessage("wrongdata");
    }

    public void logoutClient(Client client) {
        clients.remove(client);
        WebSocket.closeConnection(client.getSession());
        Log.write(Methods.createPrefix() + "User '" + client.getName() + "' logged out", false);
    }

    public void addPermission(String username, String permission) {
        if(!userPermissions.containsKey(username)) {
            final List<String> list = new ArrayList<>();
            list.add(permission);
            userPermissions.put(username, list);
        }else userPermissions.get(username).add(permission);

        if(getClient(username) != null) getClient(username).updatePermissions(userPermissions.get(username));
    }

    public boolean removePermission(String username, String permission) {
        if(!userPermissions.containsKey(username)) return false;
        if(!userPermissions.get(username).contains(permission)) return false;
        userPermissions.get(username).remove(permission);

        if(getClient(username) != null) getClient(username).updatePermissions(userPermissions.get(username));
        return true;
    }

    public boolean isLoggedIn(Session session) {
        return isLoggedIn(session.getRemoteAddress().getAddress());
    }

    public boolean isLoggedIn(InetAddress address) {
        for(Client client : clients) if(client.getAddress().equals(address)) return true;
        return false;
    }

    public boolean isLoggedIn(String username) {
        for(Client client : clients) if(client.getName().equals(username)) return true;
        return false;
    }

    public Client getClient(Session session) {
        return getClient(session.getRemoteAddress().getAddress());
    }

    public Client getClient(InetAddress address) {
        if(!isLoggedIn(address)) return null;
        for(Client client : clients) if(client.getAddress().equals(address)) return client;
        return null;
    }

    public Client getClient(String username) {
        if(!isLoggedIn(username)) return null;
        for(Client client : clients) if(client.getName().equals(username)) return client;
        return null;
    }

}
