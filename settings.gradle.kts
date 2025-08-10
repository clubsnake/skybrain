pluginManagement {
  repositories {
    gradlePluginPortal()
    google()
    mavenCentral()
    maven { url = uri("https://artifact.dji.com/repo") }
  }
}
dependencyResolutionManagement {
  repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
  repositories {
    google()
    mavenCentral()
    maven { url = uri("https://artifact.dji.com/repo") }
  }
}
rootProject.name = "skybrain"
include(":app")
