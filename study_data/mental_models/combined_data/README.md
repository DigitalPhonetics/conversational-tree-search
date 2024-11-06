## Files
1. `chat_log_full.txt`: contains the unfiltered chatlog from the system, showing the full path the policy took through the dialog tree (including nodes not shown to users) with dialog turns as they occured. The asynchronous logging means dialogs occuring at the same time are often logged interweavingly
2. `transcript.txt`: contains the filtered version of the transcript, with dialogs organized by user into system and user turns. System turns are truncated to make reading easier. Users who did not engage with the system or who quit the interaction without filling out a survey are filtered out
3. `markdown_transcript.txt`: contains the same thing as filtered transcript, but with the full system utterance including the html formatting
4. `user_log_full.txt`: contains the user consent and condition to which the user was assigned
5. `survey_log_full.txt`: Contains the user responses to the pre and post surveys (demographic information, free-response, and likert questions about expectations and impressions after the interaction, i.e., mental models, trust, usability, reliability)
6. `expectations_met.csv`: contains content analysis labels for each user response to the question 'How well were your expectations met?'
7. `system_strengths.csv`: contains content analysis labels for each user response to the question 'What could the chatbot do well?'
8. `system_weaknesses.csv`: contains content analysis labels for each user response to the question 'What could the chatbot not do well?'
9. `user_likes.csv`: contains content analysis labels for each user response to the question 'What did you like about your interaction with the chatbot?'
10. `user_dislikes.csv`: contains content analysis labels for each user response to the question 'What did you not like about your interaction with the chatbot?'
11. `mm_input.csv`: contains content analysis of user mental models of what type of input a chatbot understands before interacting with the chatbot
12. `mm_output.csv`: contains content analysis of user mental models of what type of output a chatbot can give before interacting with the chatbot
13. `mm_interaction.csv`: contains content analysis of user mental models of how an interaction with a chatbot would go before interacting with the chatbot