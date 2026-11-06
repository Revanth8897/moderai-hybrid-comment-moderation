
import shutil
import stat
import time

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OLD_DIRECTORIES = [
    ROOT / "datasets" / "processed" / "tfidf",
    ROOT / "datasets" / "processed" / "tfidf_random",
]

OLD_FILES = [
    ROOT / "models" / "tfidf_vectorizer_random.joblib",
]


def handle_permission_error(func, path, exc_info):

    """
    Handle read-only files and retry deletion.
    """

    try:
        Path(path).chmod(
            stat.S_IWRITE | stat.S_IREAD
        )

        func(path)

    except Exception:
        raise


def main():

    print("=" * 65)
    print("MODERAI TF-IDF CLEANUP RETRY")
    print("=" * 65)

    for directory in OLD_DIRECTORIES:

        if directory.exists():

            try:

                shutil.rmtree(
                    directory,
                    onerror=handle_permission_error
                )

                print("[DELETED]", directory)

            except Exception as error:

                print("[FAILED]", directory)
                print("Reason:", error)

        else:

            print("[ALREADY REMOVED]", directory)

    for file in OLD_FILES:

        if file.exists():

            try:

                file.chmod(
                    stat.S_IWRITE | stat.S_IREAD
                )

                file.unlink()

                print("[DELETED]", file)

            except Exception as error:

                print("[FAILED]", file)
                print("Reason:", error)

        else:

            print("[ALREADY REMOVED]", file)

    print("\n" + "=" * 65)
    print("CLEANUP RETRY FINISHED")
    print("=" * 65)


if __name__ == "__main__":
    main()
