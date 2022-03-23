# Boolean Information Retrieval System
Boolean Information Retrieval System with wildcard and spell check support

To run the project:
1. Clone the project.
2. Create a directory named "Data" and store all your documents and anopther directory "preprocessed_data" for the preprocessed files to be written into.
3. Run the "IR_system" python program.
4. After the program is done with preprocessing the documents, the user query is to be entered following the format as shown through an example below.
    
    > ( ( harry AND met ) OR ( NOT sally ) ) , space is to be given after every keyword, operator, bracket. 
    
    > (william) wont work as there is not gap between keyword and bracket

The output for the query will be calculated and printed out on the console.

Test cases to try out:
> Normal search: ( valeria ).  
> Correct words with AND operator: ( dissolutely AND bardolph ).  
> Correct words with OR operator: ( scarf  OR merriness ).  
> Correct words with NOT operator: ( NOT william ).  
> WIldcard: ( val*a ).  
> AND and NOT: ( Ceasar AND ( NOT william ) ).  
> WIldcard and OR: ( val*a OR bar* ).  
> Wrong Spelling: ( valer ).  



user can continuously query and then can pass "-1" when they dont wish to pass any more new queries.


## Stopword Removal and Lemmatization
![](/stop_and_lemmatize.png)

## Inverted Index
![](/inverted_index.png)

## Execution of a query
![](/querying.png)
