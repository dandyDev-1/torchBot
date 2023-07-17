import os

ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIRECTORY = os.path.join(ROOT_DIRECTORY, "database")
LOG_DIRECTORY = os.path.join(DATABASE_DIRECTORY, "logs")

APP_NAME = ROOT_DIRECTORY.split("/")[-1]


ENV_FILE = os.path.join(DATABASE_DIRECTORY, '.env')
DATABASE_LOG = os.path.join(LOG_DIRECTORY, "database.log")
INTENTS_FILE = os.path.join(DATABASE_DIRECTORY, "intents.json")
PTH_FILE = os.path.join(DATABASE_DIRECTORY, "data.pth")


def checkDirs():
    directories = [ROOT_DIRECTORY, DATABASE_DIRECTORY, LOG_DIRECTORY]
    files = [ENV_FILE, DATABASE_LOG, INTENTS_FILE]

    for Dir in directories:
        if not os.path.exists(Dir):
            os.mkdir(Dir)
            print(f"CREATED {Dir}")

    for file in files:
        if not os.path.exists(file):
            open(file, "a").close()
            print(f"CREATED {file}")


checkDirs()

