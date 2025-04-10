def my_decorator(func):
    def wrapper():
        print("Something before the function")
        func()
        print("Something before the function")
    return wrapper

@my_decorator
def say_hello():
    print("Hello !")

say_hello()