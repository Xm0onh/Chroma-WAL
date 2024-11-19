import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from pipeline import process_files
from snapshot import create_snapshot
from config import DATA_PATH

class FolderWatcher(FileSystemEventHandler):
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=30),  
        stop=stop_after_attempt(5),                         
        retry=retry_if_exception_type(Exception),           
    )
    def on_created(self, event):
        """
        Triggered when a new file is created in the `data` folder.
        """
        if not event.is_directory and event.src_path.endswith('.pdf'):
            print(f"📄 New file detected: {event.src_path}")
            process_files()
            create_snapshot()

def watch_folder():
    """
    Watch the `data` folder for changes.
    """
    event_handler = FolderWatcher()
    observer = Observer()
    observer.schedule(event_handler, DATA_PATH, recursive=False)
    observer.start()
    print(f"👀 Watching for changes in {DATA_PATH}...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    watch_folder()
