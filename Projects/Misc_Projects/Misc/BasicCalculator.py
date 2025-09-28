#function to add two numbers
def add_numbers(num1, num2):
    return num1 + num2

#function to subtract two numbers
def subtract_numbers(num1, num2):
    return num1 - num2

#function to multiply two numbers
def multiply_numbers(num1, num2):
    return num1 * num2

#function to divide two numbers
def divide_numbers(num1, num2):
    if num2 == 0:
        return "Error: Division by zero"    
    else:
        return num1 / num2

def calculator():
    print("Select operation.")
    print("1.Add")
    print("2.Subtract")
    print("3.Multiply")
    print("4.Divide")
    
    while True:
        #take input from the user
        choice = input("Enter choice(1/2/3/4): ")   

        #check if choice is one of the four options
        if choice in ('1', '2', '3', '4'):
            num1 = float(input("Enter first number: "))
            num2 = float(input("Enter second number: "))

            if choice == '1':
                print(num1, "+", num2, "=", add_numbers(num1, num2))
            elif choice == '2':
                print(num1, "-", num2, "=", subtract_numbers(num1, num2))
            elif choice == '3':
                print(num1, "*", num2, "=", multiply_numbers(num1, num2))
            elif choice == '4':
                print(num1, "/", num2, "=", divide_numbers(num1, num2))
    
        #Option to exit the loop
        next_calculation = input("Do you want to perform another calculation? (yes/no): ")
        if next_calculation.lower() != 'yes':
            break   
    print("Thank you for using the calculator.Goodbye!")

#Call the calculator function
calculator()
                
