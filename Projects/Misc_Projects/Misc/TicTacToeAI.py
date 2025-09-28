#Define the board as a list
board = [' ' for _ in range(9)] #9 spaces for the 9 positions on the board

#Function to print the board
def print_board(board):
    for row in [board[i*3:(i+1)*3] for i in range(3)]:
        print('|' + '|'.join(row) + '|')

#Function to check if a player has won
def check_winner(board, player):
    win_conditions = [
         [0, 1, 2], [3, 4, 5], [6, 7, 8], #rows
         [0, 3, 6], [1, 4, 7], [2, 5, 8], #columns
         [0, 4, 8], [2, 4, 6] #diagonals
     ]
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] == player:
            return True
    return False

#Function to check if the board is full
def is_board_full(board):
    return ' ' not in board
#function to evaluate the board for the minimax algorithm
def evaluate_board(board):
    if check_winner(board, '0'): #AI  is 0
        return 1
    elif check_winner(board, 'X'): #Human is X
        return -1
    else:
        return 0
#Minimax function to calculate the best move for the AI
def minimax(board, depth, is_maximizing):
    score = evaluate_board(board)
    if score ==1 or score == -1 or is_board_full(board):
        return score
    if is_maximizing: #AI`s turn
        best_score = -float('inf') #Initialize the best score to negative infinity
        for i in range(9):
            if board[i] == ' ':
                board[i] = '0' #AI makes a move
                score = minimax(board, depth + 1, False) #Recursively call minimax for the human player
                board[i] = ' ' #Undo the move
                best_score = max(best_score, score)
        return best_score
    else: #Human`s turn
        best_score = float('inf') #Initialize the best score to positive infinity
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'X' #Human makes a move
                score = minimax(board, depth + 1, True) #Recursively call minimax for the AI player
                board[i] = ' ' #Undo the move
                best_score = min(best_score, score)
        return best_score

#Function to find the best move for the AI
def find_best_move(board):
    best_score = -float('inf')
    best_move = -1
    for i in range(9):
        if board[i] == ' ':
            board[i] = '0' #AI makes a move
            score = minimax(board, 0, False)
            board[i] = ' '
            if score > best_score:
                best_score = score
                best_move = i
    return best_move


#Main game loop
def play_game():
    while True: #Main game loop
        print("Tic Tac Toe")    
        print_board(board)
    
        #Player move
        player_move = int(input("Enter your move (1-9): ")) - 1
        if board[player_move] != ' ': #Check if the move is valid
            print("Invalid move. Try again.")
            continue
        board[player_move] = 'X' #Human makes a move
        print("You moved to position", player_move + 1)

        #Check if a player won
        if check_winner(board, 'X'):
            print_board(board)
            print("You win!")
            break
        #Check for draw
        if is_board_full(board):
            print_board(board)
            print("It's a draw!")
            break
        #AI move
        print("AI is making its move...")
        ai_move = find_best_move(board)
        board[ai_move] = '0'
       

        #Check if AI won
        if check_winner(board, '0'):
            print_board(board)
            print("AI wins!")
            break
        #Check for draw
        if is_board_full(board):
            print_board(board)
            print("It's a draw!")
            break   


#Start the game
play_game()


        
        

        
        
