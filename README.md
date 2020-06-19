# REDDIT_leadingIndiaAI
Reddit is a website which comprises user-generated content including photos, videos, links, and text-based posts.
Our task was to create a model  for Topic Modeling . It is the task of using unsupervised learning to extract the main topics 
(represented as a set of words) that occur in our reddit posts.
We worked on the reddit/india/politics  dataset (5 columns and 93504 rows ) and performed “Pre Processing” on it .

We implemented the different techniques of NLP to analyse various ways in which reddit data could be analysed. 
We were able to decide the domain on which we wanted to work through preprocessing of data. 
Through EDA we got the idea regarding which words , bi grams , tri grams etc were more common(Modi and Prime Minister  were in clear lead) 
and the type of length , Number of words in a post.
We were able to analyse the time frame where more number of posts were made. 
The most  optimum output from the LDA model is a 6 topics each categorized by a series of top 10 words. 
LDA doesn’t give a topic name to those categories and it is for us humans to interpret them. 
The model runs very quickly , we were able to extract common topics in a few seconds . 
Limitations :
Our model assumes that there are distinct topics in the data set. Since our data set contains only data from one particular domain-’Politics’ , the result might not be very easily interpretable. Choosing best parameters depends a lot  on the human perception

#cleaning file- the code for cleaning the dataset and saving only those rows and columns which we are going to use for further preprocessing , eda and modeling. Its output is saved in form of given "reddit" csv file.

#eda- The code for preprocessing,eda(except year/month/day distribution) and modeling of data.

#eda_time_distribution-year/month/day vs frequency distribution

