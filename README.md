# Fetch-Wont-Happen
Boolean Information Retrieval System with wildcard and spell check support

To run the project:
1. Clone the project
2. Create a directory named "Data" and store all your documents and anopther directory "preprocessed_data" for the preprocessed files to be written into
3. Run the "IR_system" python program
4. After the program is done with preprocessing the documents, the user query is to be entered. This has a particular format:
    For ex:  ( ( harry AND met ) OR ( NOT sally ) ) 
    --> space is to be given after every keyword,Operator,bracket <--
The output for the query will be calculated and printed out on the console.

user can continuously query and then can pass "-1" when they dont wish to pass any more new queries