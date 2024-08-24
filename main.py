import sys
import task1
import task2
import task3


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py [task1|task2|task3]")
        sys.exit(1)
    
    task_name = sys.argv[1]
    
    if task_name == 'task1':
        task1.task1()
    elif task_name == 'task2':
        task2.task2()
    elif task_name == 'task3':
        task3.task3()
    else:
        print("Invalid task name. Please use 'task1', 'task2', or 'task3'.")
        sys.exit(1)

if __name__ == "__main__":
    main()