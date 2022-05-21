package net.enumtype.homesystem.plugin;

import net.enumtype.homesystem.plugin.events.Event;
import net.enumtype.homesystem.plugin.events.EventHandler;
import net.enumtype.homesystem.plugin.events.Listener;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.*;
import java.util.jar.*;

public class PluginManager {

    private final List<Plugin> plugins = new ArrayList<>();
    private final List<Listener> listeners = new ArrayList<>();
    private final Map<String, List<Command>> commands = new HashMap<>();

    public PluginManager() {
        try {
            loadPlugins(new File("plugins"));
        }catch(Exception e) {
            e.printStackTrace();
        }
    }

    public void loadPlugins(File directory) throws Exception {
        if(!directory.exists() && !directory.mkdirs()) throw new PluginException("Cannot load plugins!");
        final File[] files = directory.listFiles(((dir, name) -> name.endsWith(".jar")));
        if(files == null || files.length < 1) return;
        for(File file : files) {
            loadPlugin(file);
        }
    }

    public void unloadPlugins() {
        for(Plugin plugin : plugins) {
            plugin.stop();
        }

        plugins.clear();
    }

    private void loadPlugin(File file) throws Exception {
        final JarFile jar = new JarFile(file);
        final URLClassLoader loader = new URLClassLoader(new URL[] {new URL("jar:" + file.toURI().toURL() + "!/")});
        final URL pluginConfig = loader.getResource("plugin.cfg");
        if(pluginConfig == null) throw new PluginException("No plugin.cfg found in " + file.getName());

        final InputStream stream = pluginConfig.openStream();


        if(stream == null) return;
        final BufferedReader in = new BufferedReader(new InputStreamReader(stream));

        String line;
        String main = null;
        String name = null;
        String version = null;
        String author = null;

        while((line = in.readLine()) != null) {
            final String[] args = line.split(": ");

            if(args.length < 2) throw new PluginException("Invalid plugin.cfg syntax in " + file.getName()  + "!");

            switch(args[0].toLowerCase()) {
                case "name":
                    name = args[1];
                    break;
                case "version":
                    version = args[1];
                    break;
                case "main":
                    main = args[1];
                    break;
                case "author":
                    author = args[1];
            }
        }

        if(name == null || version == null || main == null) throw new PluginException("Missing plugin information in " +
                                                                file.getName() + "!");

        Class<?> clazz = loader.loadClass(main);

        if(!clazz.getSuperclass().getName().equals(Plugin.class.getName())) return; //TODO: throw exception
        final Plugin plugin = (Plugin) clazz.getDeclaredConstructor().newInstance();
        plugin.setName(name);
        plugin.setVersion(version);
        if(author != null) plugin.setAuthor(author);
        plugin.load();
        plugins.add(plugin);

        in.close();
        loader.close();
        jar.close();
    }

    public void registerListener(Listener listener) {
        listeners.add(listener);
    }

    public void registerCommand(String name, Command command) {
        if(!commands.containsKey(name)) commands.put(name, new ArrayList<>());
        if(!commands.get(name).contains(command)) commands.get(name).add(command);
    }

    public void triggerEvent(Event event) {
        try {
            for(Listener listener : listeners) {
                for(Method method : listener.getClass().getDeclaredMethods()) {
                    if(!method.isAnnotationPresent(EventHandler.class)) continue;
                    if(!(new ArrayList<>(Arrays.asList(method.getParameterTypes()))).contains(event.getClass())) continue;
                    method.setAccessible(true);
                    method.invoke(listener, event);
                }
            }
        }catch(Exception e) {
            e.printStackTrace();
        }
    }

    public void triggerCommand(String name, String[] args) {
        if(!commands.containsKey(name)) return;
        commands.get(name).forEach(command -> command.execute(name, args));
    }

}

class PluginException extends Exception {

    public PluginException(String message) {
        super(message);
    }

}