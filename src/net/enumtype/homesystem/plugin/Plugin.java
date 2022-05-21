package net.enumtype.homesystem.plugin;

public abstract class Plugin {

    public String name = "";
    public String version = "";
    public String author = "";

    public abstract void start();
    public abstract void stop();

    public void load() {
        System.out.println("Loading " + getName() + " version " + getVersion() + "...");
        start();
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setVersion(String version) {
        this.version = version;
    }

    public void setAuthor(String author) {
        this.author = author;
    }

    public String getName() {
        return name;
    }

    public String getVersion() {
        return version;
    }

    public String getAuthor() {
        return author;
    }

}
