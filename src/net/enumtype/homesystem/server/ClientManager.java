package net.enumtype.homesystem.server;

import net.enumtype.homesystem.Main;
import net.enumtype.homesystem.utils.Log;
import org.eclipse.jetty.websocket.api.Session;
import org.yaml.snakeyaml.DumperOptions;
import org.yaml.snakeyaml.Yaml;

import java.io.*;
import java.net.InetAddress;
import java.util.*;

public class ClientManager {

    private final Map<String, String> userData = new HashMap<>();
    private final Map<String, List<String>> userPermissions = new HashMap<>();
    private final List<Client> clients = new ArrayList<>();
    private final Log log;

    public ClientManager() {
        this.log = Main.getLog();

        try {
            load();
        }catch(IOException e) {
            log.writeError(e);
        }
    }

    public void load() throws IOException {
        loadUserData();
        loadUserPerm();
    }

    public void loadUserData() throws IOException {
        if(!userData.isEmpty()) userData.clear();

        File file = new File("User-Data");
        File dataFile = new File(file + "//user.data");
        if(!file.exists()) if(!file.mkdirs()) throw new IOException("Cannot create directories");
        if(!dataFile.exists()) {
            registerUser("a", "a");
            removeUser("a");
        }

        BufferedReader reader = new BufferedReader(new FileReader(file + "//user.data"));
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

    public boolean registerUser(String username, String password) throws IOException {
        if(userData.containsKey(username)) return false;

        final PrintWriter out = new PrintWriter(new FileWriter("User-Data//user.data", true), true);
        out.write("Name: " + username + "\r\n");
        out.write("Password: " + password + "\r\n");
        out.close();

        userData.put(username, password);
        userPermissions.put(username, new ArrayList<>());
        writeUserPerm(false);
        return true;
    }

    public boolean removeUser(String username) throws IOException {
        if(!userData.containsKey(username)) return false;

        File data = new File("User-Data//user.data");
        File newData = new File("User-Data//user.data.2");
        BufferedReader reader = new BufferedReader(new FileReader(data));
        BufferedWriter writer = new BufferedWriter(new FileWriter(newData));
        String line;

        while((line = reader.readLine()) != null) {
            if(!line.equals("Name: " + username) && !line.equals("Password: " + userData.get(username)) && !line.equals("\r\n"))
                writer.write(line + "\r\n");
        }

        reader.close();
        writer.close();
        if(!data.delete()) throw new IOException("Cannot delete file!");
        if(!newData.renameTo(data)) throw new IOException("Cannot rename file!");

        userData.remove(username);
        userPermissions.remove(username);
        if(isLoggedIn(username)) getClient(username).logout();
        writeUserPerm(false);

        return true;
    }

    @SuppressWarnings("unchecked")
    public void loadUserPerm() throws IOException {
        File file = new File("User-Data//Permissions.yml");

        if(!userPermissions.isEmpty()) userPermissions.clear();
        if(!file.exists()) {
            log.write("Creating Permissions.yml...");
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

        log.write("Loading Permissions.yml...");
        Yaml yaml = new Yaml();
        FileInputStream io = new FileInputStream(new File("User-Data//Permissions.yml"));

        Map<Object, List<String>> list = (Map<Object, List<String>>) yaml.load(io);

        list.keySet().forEach(user -> userPermissions.put(user.toString(), list.get(user)));
    }

    public void writeUserPerm(boolean logging) throws IOException {
        if(logging) log.write("Saving Permissions.yml...");
        for(String user : userPermissions.keySet()) if(!userData.containsKey(user)) userPermissions.remove(user);

        DumperOptions options = new DumperOptions();
        options.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK);
        options.setPrettyFlow(true);

        Yaml yaml = new Yaml(options);
        FileWriter writer = new FileWriter("User-Data//Permissions.yml");
        yaml.dump(userPermissions, writer);
    }

    public void addPermission(String username, String permission) {
        try {
            if(!userPermissions.containsKey(username)) {
                final List<String> list = new ArrayList<>();
                list.add(permission);
                userPermissions.put(username, list);
            }else userPermissions.get(username).add(permission);

            if(getClient(username) != null) getClient(username).updatePermissions(userPermissions.get(username));
            writeUserPerm(false);
        }catch(IOException e) {
            log.writeError(e);
        }
    }

    public void addClient(Client client) {
        clients.add(client);
        client.updatePermissions(userPermissions.containsKey(client.getName()) ?
                userPermissions.get(client.getName()) : new ArrayList<>());
    }

    public void removeClient(Client client) {
        clients.remove(client);
    }

    public void logoutAll() {
        final List<Client> list = new ArrayList<>(clients);
        list.forEach(Client::logout);
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

    public boolean isChangingConnection(Session session) {
        if(!isLoggedIn(session)) return false;

        return getClient(session).isChangingConnection();
    }

    public boolean isDataCorrect(String username, String password) {
        if(!userData.containsKey(username)) return false;
        return userData.get(username).equals(password);
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
