/***********************************************************************************************

██████╗ ███████╗ █████╗ ██████╗     ███╗   ███╗███████╗██╗
██╔══██╗██╔════╝██╔══██╗██╔══██╗    ████╗ ████║██╔════╝██║
██████╔╝█████╗  ███████║██║  ██║    ██╔████╔██║█████╗  ██║
██╔══██╗██╔══╝  ██╔══██║██║  ██║    ██║╚██╔╝██║██╔══╝  ╚═╝
██║  ██║███████╗██║  ██║██████╔╝    ██║ ╚═╝ ██║███████╗██╗
╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝     ╚═╝     ╚═╝╚══════╝╚═╝

************************************************************************************************
Copyright(c): 2024 Snowflake Inc. All rights reserved.

This script has been developed to help you understand how to use Snowflake AI & ML features 
within the Snowsight UI and in Streamlit. To use this script follow the readme below

Disclaimer: The code below is provided "as is" and provided with no warranties, support or 
liabilities, use at your own discretion! In no event shall Snowflake or contributors be liable 
for any direct, indirect, incidental, special, exemplary, or consequential damages sustained by 
you or a third party, however caused, arising in any way out of the use of this sample code.

To complete the demo you will need to:
1. Setup a new Snowflake Account and prepare the demo environment as explained in the 
2024-07 LSUG - Setup Script.sql.

2. Create a new SQL Worksheet and copy and paste all the code in this SQL script into it.



-- FURTHER READING:
-- 📖 https://quickstarts.snowflake.com/guide/getting-started-with-snowflake-cortex-ml-forecasting-and-classification/index.html

***********************************************************************************************/



-- EXPLORATORY DATA ANALYSIS
-- ======================================================================================
-- Before building our model, let's first visualize our data to get a feel for what it 
-- looks like and get a sense of what variables we will be working with.

use schema UCKFIELD_CINEMA.EPOS;



-- Create a new view and add some simple date based features our ML model can use later.

create or replace view TICKET_SALES_ENRICHED as 
    select 
        TS.* exclude ( TICKET_TYPE, TICKET_DATE ),
        DD.*,
        
    from TICKET_SALES as TS

        right join PUBLIC.DATE_DIM as DD
            on TS.TICKET_DATE = DD.DATE_VALUE
            and TS.TICKET_TYPE = DD.TICKET_TYPE

    order by DD.DATE_VALUE asc, DD.TICKET_TYPE asc
;




-- Let's quickly explore the data to see what we're working with, the data is very peaky
-- which corresponds to big increases in ticket sales over the weekends, then big blocks
-- of increases over school holiday periods.
select * 
from TICKET_SALES_ENRICHED 
where 1=1
    and DATE_VALUE >= dateadd( day, -60, current_date() ) 
;





----------------------------------------------------------------------------------------
-- FORECASTING ON A SINGLE TIME SERIES
-- 📖 https://docs.snowflake.com/en/user-guide/ml-functions/forecasting
-- 📖 https://docs.snowflake.com/sql-reference/classes/forecast/commands/create-forecast



-- 🎬 Preparing Training Data & Training the Model:
-- Let's try and predict the number of tickets purchased by our 
-- SENIOR customers. To do that we'll need a training dataset which is just the ticket 
-- sales for seniors.

create or replace view TICKET_SALES_TRAINING_SENIORS as

    select 
        DATE_VALUE::timestamp_ntz as ML_DATE_POINT,
        sum( QTY ) as QTY,
        
    from TICKET_SALES_ENRICHED

    where 1=1
        and TICKET_TYPE = 'Senior'
        and QTY is not null

    group by all
;



-- 💬 Let's create the model and train it. A Warehouse provides the compute resources for training and using 
--    the model. The training step, the most time-consuming and memory-intensive part of the process. There are 
--    two key factors to keep in mind when choosing a warehouse:
--    1️⃣ The number of ROWS and COLUMNS your data contains.
--    2️⃣ The number of distinct SERIES your data contains.

--    Use the following rules of thumb to choose your warehouse:
--    1️⃣ If you are training on a longer time series (> 5 million rows) or on many columns (many features), 
--       consider upgrading to Snowpark-optimized warehouses.
--    2️⃣ If you are training on many series, size up. The forecasting function distributes model training 
--       across all available nodes in your warehouse when you are training for multiple series at once.

--    As a rough estimate, training time is proportional to the number of rows in your time series. For 
--    example, on a XS standard warehouse, with evaluation turned off (CONFIG_OBJECT => {'evaluate': False}), 
--    training on a 100,000-row dataset takes about 400 seconds. Training on a 1,000,000-row dataset takes 
--    about 850 seconds. With evaluation turned on, training time increases roughly linearly by the number of 
--    splits used.

-- 📖 https://docs.snowflake.com/en/user-guide/ml-functions/forecasting#warehouse-selection

create or replace SNOWFLAKE.ML.FORECAST TICKET_SALES_SENIORS(
    
    input_data => table( TICKET_SALES_TRAINING_SENIORS ),
    timestamp_colname => 'ML_DATE_POINT',
    target_colname => 'QTY',

    -- 💬 You have two choices to use in the "method" argument of the config_object. "best" uses an ensemble 
    --    of models including Prophet, ARIMA , Exponential Smoothing, and a gradient boosting machine (GBM) 
    --    based algorithm. "fast" uses a GBM based algorithm, which is faster but may not be as accurate.
    --    Use 'fast' when your training data has 10,000 or more individual series.
    
    config_object => {
        'method': 'fast', -- Valid values are 'best' or 'fast'.
        'frequency': '1 day',
        'evaluate': True
    }
    
);




-- 🎬 Testing the Model:
-- Now we have created and trained our model, lets test it!
select 
    'Actual' as "Actual or Forecast",
    DATE_VALUE_TS::date as "Date",
    QTY as "Quantity",
    null as "Forecast",
    null as "Lower Bound",
    null as "Upper Bound",
    
from TICKET_SALES_TRAINING_SENIORS 

where 1=1
    and DATE_VALUE_TS >= '2024-01-01 00:00:00'
    and QTY is not null

union all

-- This is where the model we created above is used to predict the future!
select 
    'Forecast',
    TS::date,
    null,
    FORECAST,
    LOWER_BOUND,
    UPPER_BOUND,
    
from table( TICKET_SALES_SENIORS!FORECAST( 
    forecasting_periods => 60,
    config_object => {'prediction_interval': 0.25}
) );



-- But what actually happened? Which features are most important for the model? The following 
-- helper functions allw you to assess your model performance, understand which features are 
-- most impactful to your model, and to help you debug the training process.
-- During testing fast gave a MAPE of about 0.558 and best gave a MAPE of 0.162.
call TICKET_SALES_SENIORS!SHOW_EVALUATION_METRICS();



-- You can list all the models using the show command, we should have just the one.
show SNOWFLAKE.ML.FORECAST;









----------------------------------------------------------------------------------------
-- EXAMPLE 2: FORECASTING USING A MULTIPLE TIME SERIES WITH EXOGENOUS VARIABLES
-- 📖 https://docs.snowflake.com/en/user-guide/ml-functions/forecasting
-- 📖 https://docs.snowflake.com/sql-reference/classes/forecast/commands/create-forecast


-- 🎬 Preparing Training Data & Training the Model:
-- Let's try and predict the number of tickets purchased for ALL customers. Now we have 
-- an idea of what the data looks like, we can split it into data used for TRAINING and
-- data used for PREDICTING.

create or replace view TICKET_SALES_TRAINING as 

    select 
        to_timestamp_ntz( DATE_VALUE ) as ML_DATE_POINT,
        * EXCLUDE ( DATE_VALUE, CINEMA_ID, TOTAL_AMOUNT ),
        
    from TICKET_SALES_ENRICHED

    where 1=1
        -- The training data is from 2 years to 30 days ago.
        and DATE_VALUE between dateadd( year, -2, current_date() ) and dateadd( day, -30, current_date() )

    order by ML_DATE_POINT asc, TICKET_TYPE asc
;

select * from TICKET_SALES_TRAINING order by ML_DATE_POINT desc limit 10;



-- Create a view which will be our basis for forecasting future ticket sales.
create or replace view TICKET_SALES_FORECAST as 

    select 
        to_timestamp_ntz( DATE_VALUE ) as ML_DATE_POINT,
        * EXCLUDE ( DATE_VALUE, CINEMA_ID, TOTAL_AMOUNT ),
        
    from TICKET_SALES_ENRICHED

    where 1=1
        -- We'll limit the view to looking back 29 days and into the future 30 days, making sure
        -- we don't overlap with the training data.
        and DATE_VALUE between dateadd( day, -29, current_date() ) and dateadd( day, 30, current_date() )

    order by DATE_VALUE asc, TICKET_TYPE asc
;

select * from TICKET_SALES_FORECAST order by ML_DATE_POINT desc;



-- Create your model.
create or replace SNOWFLAKE.ML.FORECAST TICKET_SALES_FORECAST(

    input_data => system$reference('VIEW', 'TICKET_SALES_TRAINING'),
    series_colname => 'TICKET_TYPE',
    timestamp_colname => 'ML_DATE_POINT',
    target_colname => 'QTY',
    
    config_object => { 
        'method': 'fast',
        'evaluate': false , 
        'on_error': 'SKIP'
    }
    
);



-- 🎬 Testing the Model:
-- Now we have created and trained our model, lets test it! In this case
-- we'll save the results to a table.
begin

    -- This is the step that creates your predictions.
    call TICKET_SALES_FORECAST!FORECAST(
    
        input_data => SYSTEM$REFERENCE('VIEW', 'TICKET_SALES_FORECAST'),
        series_colname => 'TICKET_TYPE',
        timestamp_colname => 'ML_DATE_POINT',
        config_object => {
            'prediction_interval': 0.95
        }
        
    );
    
    -- These steps store your predictions to a table.
    let x := SQLID;
    create or replace table TICKET_SALES_PREDICTIONS as select * from table(result_scan(:x));
    
end;

select * from TICKET_SALES_PREDICTIONS order by TS desc;



-- Union your predictions with your historical data, then view the results in a chart.
select 
    TICKET_TYPE, 
    DATE_VALUE, 
    QTY AS ACTUAL, 
    null AS FORECAST, NULL AS LOWER_BOUND, NULL AS UPPER_BOUND
    
    FROM TICKET_SALES_ENRICHED
    
    WHERE 
        DATE_VALUE between '2025-01-01' and dateadd( day, 0, current_date() )
        
UNION ALL

SELECT 
    replace(series, '"', '') as TICKET_TYPE, 
    ts as ML_DATE_POINT, 
    null as ACTUAL, FORECAST, LOWER_BOUND, UPPER_BOUND
    
    from TICKET_SALES_PREDICTIONS
    
    where 
        ML_DATE_POINT >= dateadd( day, -7, current_date() )
;




-- Or we could use ML Studio...









/*********************************************************************************************************

██╗     ██╗     ███╗   ███╗    ███████╗██╗   ██╗███╗   ██╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗███████╗
██║     ██║     ████╗ ████║    ██╔════╝██║   ██║████╗  ██║██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
██║     ██║     ██╔████╔██║    █████╗  ██║   ██║██╔██╗ ██║██║        ██║   ██║██║   ██║██╔██╗ ██║███████╗
██║     ██║     ██║╚██╔╝██║    ██╔══╝  ██║   ██║██║╚██╗██║██║        ██║   ██║██║   ██║██║╚██╗██║╚════██║
███████╗███████╗██║ ╚═╝ ██║    ██║     ╚██████╔╝██║ ╚████║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║███████║
╚══════╝╚══════╝╚═╝     ╚═╝    ╚═╝      ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝
                                                                                                         
*********************************************************************************************************/


-- Snowflake Cortex gives you instant access to industry-leading large language models (LLMs) trained by 
-- researchers at companies like Mistral, Reka, Meta, and Google, including Snowflake Arctic, an open 
-- enterprise-grade model developed by Snowflake.

-- Snowflake Cortex features are provided as SQL functions and are also available in Python. The 
-- available functions are summarized below.

    -- SENTIMENT: Returns a sentiment score, from -1 to 1, representing the detected positive or negative 
    -- sentiment of the given text.
    -- COMPLETE: Given a prompt, returns a response that completes the prompt. This function accepts 
    -- either a single prompt or a conversation with multiple prompts and responses.
    -- EXTRACT_ANSWER: Given a question and unstructured data, returns the answer to the question if it 
    -- can be found in the data.
    -- SUMMARIZE: Returns a summary of the given text.
    -- TRANSLATE: Translates given text from any supported language to any other.
    -- EMBED_TEXT_768: Given a piece of text, returns a vector embedding of 768 dimensions that represents 
    -- that text.
    -- EMBED_TEXT_1024: Given a piece of text, returns a vector embedding of 1024 dimensions that represents 
    -- that text.



    
-- EXAMPLE 1: TESTING OUT BASIC LLM FUNCTIONS

-- Lets test SENTIMENT out
select SNOWFLAKE.CORTEX.SENTIMENT(
    'I really hate the book The Road. I loved No Country for Old Men, and thought I\'d enjoy The Road, but its a truely terrifying, depressing book!'
) as SENTIMENT;


select SNOWFLAKE.CORTEX.SENTIMENT(
    'I loved the book The Road, what a wonderful, happy story!'
) as SENTIMENT;



-- Lets test COMPLETE
select SNOWFLAKE.CORTEX.COMPLETE(
    'mistral-large',
    'Critique this review in bullet points: <review>I''m currently a freshman in college, but I read this book during the second semester of my sophomore year of high school, which is when the coronavirus kind of started getting big.  Our teacher offered up six books, and everyone in the class got to choose which one they wanted to read.  I was the ONLY person to pick this novel, and I''m glad that I did.  It''s a great book and I could draw a lot of comparisons between it and the world I was living in three years ago. It''s a dark, and depressing novel, but there are glimmers of light sprinkled within it. The bond between father and son is a strong one, it made me think of my dad, and the sacrifices he made for me. It''s a good book.</review>'
) as COMPLETED;



-- Lets test EXTRACT_ANSWER
set QUESTION = 'I''m currently a freshman in college. I read The Road during the second semester of my sophomore year of high school during coronavirus. I was the ONLY person to pick this novel, and I''m glad that I did. The bond between father and son is a strong one.';

select SNOWFLAKE.CORTEX.EXTRACT_ANSWER( $QUESTION, 'What is the book called?' ) as EXTRACT_ANSWER;

select SNOWFLAKE.CORTEX.EXTRACT_ANSWER( $QUESTION, 'What year in high school were they in?' ) as EXTRACT_ANSWER;

select SNOWFLAKE.CORTEX.EXTRACT_ANSWER( $QUESTION, 'Were they glad they chose the book?' ) as EXTRACT_ANSWER;



-- Lets test TRANSLATE
select SNOWFLAKE.CORTEX.TRANSLATE(
    'I''m currently a freshman in college, but I read this book during the second semester of my sophomore year of high school, which is when the coronavirus kind of started getting big.  Our teacher offered up six books, and everyone in the class got to choose which one they wanted to read.  I was the ONLY person to pick this novel, and I''m glad that I did.  It''s a great book and I could draw a lot of comparisons between it and the world I was living in three years ago.  It''s a dark, and depressing novel, but there are glimmers of light sprinkled within it.  The bond between father and son is a strong one, it made me think of my dad, and the sacrifices he made for me. It''s a good book.',
    'en',
    'fr'
) as TRANSLATED_TEXT;



-- Lets test SUMMARIZE
select SNOWFLAKE.CORTEX.SUMMARIZE(
    'I''m currently a freshman in college, but I read this book during the second semester of my sophomore year of high school, which is when the coronavirus kind of started getting big.  Our teacher offered up six books, and everyone in the class got to choose which one they wanted to read.  I was the ONLY person to pick this novel, and I''m glad that I did.  It''s a great book and I could draw a lot of comparisons between it and the world I was living in three years ago.  It''s a dark, and depressing novel, but there are glimmers of light sprinkled within it.  The bond between father and son is a strong one, it made me think of my dad, and the sacrifices he made for me. It''s a good book.'
) as SUMMARIZED_TEXT,
len( SUMMARIZED_TEXT );




-- Now we know how the basics work, let's save us a whole bunch of work and translate, summarise and 
-- perform sentiment analysis on the reviews. First we need to make sure we have an English language
-- version of the reviews.
update UCKFIELD_CINEMA.CRM.CUSTOMER_REVIEWS
    set 
        REVIEW_EN = SNOWFLAKE.CORTEX.TRANSLATE( REVIEW, LANG, 'en' )
    where
        NAME = NAME
;



update UCKFIELD_CINEMA.CRM.CUSTOMER_REVIEWS
    set 
        SENTIMENT = SNOWFLAKE.CORTEX.SENTIMENT( REVIEW_EN ),
        REVIEW_EN_SUMMARY = SNOWFLAKE.CORTEX.SUMMARIZE( REVIEW_EN )
    where
        NAME = NAME
;

select * from UCKFIELD_CINEMA.CRM.CUSTOMER_REVIEWS;









-- EXAMPLE 2: BUILDING A STREAMLIT APP
-- OK lets swap to Streamlit and get building the app!
-- To use the examples below, open up a new browser tab and create a new Streamlit application.
-- Delete all the auto-generated code, then copy and paste each section of code below, one at a time
--  so you can see the app being built up over time.

-- ***** PLEASE NOTE: Don't copy the lines that are commented out using SQL syntax, i.e. lines 
--                    starting with -- or // or /* or */

-- ------------------------
-- STEP 1:
/*
# Import python packages
import streamlit as st
import numpy as np
import json
from snowflake.snowpark import functions as spf
from snowflake.snowpark.context import get_active_session



# Get the current credentials
session = get_active_session()



# Page header.
st.title( 'AI Driven Cinema!' )



# Get the reviews and strip out the single quotes as they break SQL and 
# we need it for Cortex later.
reviews_df = session.table('UCKFIELD_CINEMA.CRM.CUSTOMER_REVIEWS').to_pandas()
reviews_df['CORTEX_SAFE_REVIEW'] = reviews_df['REVIEW'].str.replace( "'", "''" )



# Show what we got.
st.dataframe( reviews_df )
*/






-- ------------------------
-- STEP 2:
/*
# Everyone loves emojis, add some based on the sentiment then create the title which
# will be used to select the review.
sentiment_conditions = [ reviews_df['SENTIMENT'] > 0.5, reviews_df['SENTIMENT'] > 0.0 ]
sentiment_outputs = [ ' 😊 ', ' 🤨 ' ]
flag_conditions = [ reviews_df['LANG'] == 'en', reviews_df['LANG'] == 'fr' ]
flag_outputs = [ ' 🇬🇧 ', ' 🇫🇷 ' ]
reviews_df['REVIEW_DATE_STR'] = reviews_df['REVIEW_DATE'].astype(str)



# Add a column to the reviews_df which concatenates a few things and puts some emjois in.
reviews_df['REVIEW_TITLE'] = \
    reviews_df['REVIEW_DATE_STR'] + ' | ' + \
    reviews_df['REVIEW'].str[:40] + \
    ' ... ' + \
    np.select( sentiment_conditions, sentiment_outputs, ' 🤬 ' ) + \
    np.select( flag_conditions, flag_outputs, ' 🏴 ' )



# Show what we got.
st.dataframe( reviews_df['REVIEW_TITLE'] )
*/






-- ------------------------
-- STEP 3:
/*
# Create a drop down allowing the user to select a review, once selected put into a variable.
options_df = reviews_df['REVIEW_TITLE']

selected_review = st.selectbox( 
    label = 'Which review would you like to look at?',
    options = range( len( options_df ) ),
    format_func = lambda x: options_df[x],
    key = 'selected_review'
)
*/



-- ------------------------
-- STEP 4: START DISPLAYING THE REVIEW DETAILS
/*
# Pull out the values from the dataframe into individual variables.
review = reviews_df.iloc[selected_review]['REVIEW']
review_summary = reviews_df.iloc[selected_review]['REVIEW_EN_SUMMARY']
review_sentiment = reviews_df.iloc[selected_review]['SENTIMENT']
review_author = reviews_df.iloc[selected_review]['NAME']
review_email = reviews_df.iloc[selected_review]['EMAIL']
review_en = reviews_df.iloc[selected_review]['REVIEW_EN']
review_date = reviews_df.iloc[selected_review]['REVIEW_DATE']
review_has_offer = True if reviews_df.iloc[selected_review]['OFFER'] is not None else False
cortex_safe_review = reviews_df.iloc[selected_review]['CORTEX_SAFE_REVIEW']


# Display the review.
st.divider()
st.header( review_author + ' (' + review_email   + ')' )
st.metric( label='Sentiment Score', value="{:,.2f}".format(float(review_sentiment)) )
st.subheader( 'Review Summary' )
st.write( review_summary )
st.subheader( 'The Review in English' )
st.write( review_en )
st.subheader( 'The Original Review' )
st.write( review )
*/



-- ------------------------
-- STEP 5:
/*
# Lets generate some special offers!
if review_has_offer == False:
    def create_prompt( the_review ):
      
        prompt = f"""
               Generate an email from the cinema to the customer explaining an offer for 10% discounted 
               food if the sentiment of the review was happy and 25% if the sentiment was unhappy.
               Only generate one offer.
               The review is between the <review> and </review> tags.
               If you don´t have the information just say so.
               You don't need to put the actual review in the email.
               The offer code should be {review_date} in unix epoch time format, prefixed with a random code no more than four characters long.
               To use the offer code the customer should present this email at reception.
               The offer code should be wrapped in <>.
               Add emojis to make it exciting if you can.
               
               <review>  
               {the_review}
               </review>
               Answer: 
               """
    
        return prompt
    
    
    
    def ask_cortex( review_en ):
    
        prompt = create_prompt( review_en )
        cmd = 'select snowflake.cortex.complete(?, ?) as response'
        
        # df_response = session.sql( cmd, params=[st.session_state.model_name, prompt]).collect()
        df_response = session.sql( cmd, params=[model_name, prompt]).collect()
        
        return df_response
    
    
    
    
    
    st.divider()
    st.header( 'Generate a Personal Offer' )
    st.write( 'Choose your LLM then simply click Go!' )
    
    model_name = st.selectbox( 'Select your model:', session.table('UCKFIELD_CINEMA.PUBLIC.LLM') )
    
    if st.button("Generate Offer! ❄️"):
        question = str( review_en ).replace("'","")
        
        response = ask_cortex( question )
        
        st.write( response[0].RESPONSE )
    
        reviews_table = session.table("UCKFIELD_CINEMA.CRM.CUSTOMER_REVIEWS")
        reviews_table.update({"OFFER": response[0].RESPONSE}, reviews_table["NAME"] == review_author )

else:
    st.info('This review has already had an offer generated!', icon="ℹ️")


# Debug
with st.expander( 'Debug' ):
    st.write( st.session_state )
*/




-- Just for fun, you can try generating an offer for each review by using the code below.
alter warehouse COMPUTE_WH set warehouse_size = MEDIUM;

select 
    RVW.NAME,
    RVW.EMAIL,
    RVW.SENTIMENT,
    LLM.NAME,
    SNOWFLAKE.CORTEX.COMPLETE(
        LLM.NAME,
        concat( 
            'Generate an email from the cinema to the customer explaining an offer for 10% discounted food if the sentiment of the review was happy and 25% if the sentiment was unhappy. Only generate one offer.:<review>', 
            RVW.REVIEW_EN,
            '</review>'
        )
    ) as COMPLETE

from UCKFIELD_CINEMA.CRM.CUSTOMER_REVIEWS as RVW
    
    join UCKFIELD_CINEMA.PUBLIC.LLM as LLM
        on LLM.CORTEX_FUNCTION = 'COMPLETE'

where 1=1
    and RVW.NAME in ( 'Dynamo Snare', 'Chronotrapper' )
;



















/***********************************************************************************************

██████╗ ███████╗ █████╗ ██████╗     ███╗   ███╗███████╗██╗
██╔══██╗██╔════╝██╔══██╗██╔══██╗    ████╗ ████║██╔════╝██║
██████╔╝█████╗  ███████║██║  ██║    ██╔████╔██║█████╗  ██║
██╔══██╗██╔══╝  ██╔══██║██║  ██║    ██║╚██╔╝██║██╔══╝  ╚═╝
██║  ██║███████╗██║  ██║██████╔╝    ██║ ╚═╝ ██║███████╗██╗
╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝     ╚═╝     ╚═╝╚══════╝╚═╝

************************************************************************************************
Copyright(c): 2024 Snowflake Inc. All rights reserved.

This script has been developed to help you understand how to use Snowflake AI & ML features 
within the Snowsight UI and in Streamlit. To use this script follow the readme below

Disclaimer: The code below is provided "as is" and provided with no warranties, support or 
liabilities, use at your own discretion! In no event shall Snowflake or contributors be liable 
for any direct, indirect, incidental, special, exemplary, or consequential damages sustained by 
you or a third party, however caused, arising in any way out of the use of this sample code.

To complete the demo you will need to:
1. Create your Snowflake Trial Account:
- Go to https://signup.snowflake.com/
- Sign up for an ENTERPRISE trial in AWS Ireland.
- Click on the activation link in the email sent to you.
- Set your username and password.

2. Setup the demo environment:
- Create a new SQL Worksheet.
- Copy and paste all the code in this SQL script into the Worksheet.
- Run all the statements below.

***********************************************************************************************/

-- We're not worried about permissions so ACCOUNTADMIN is OK for this, but don't do it in 
-- real life!
use role ACCOUNTADMIN;
use warehouse COMPUTE_WH;



-- An XSMALL will do, also don't need a 10 min auto-suspend which is the default for trials.
alter warehouse COMPUTE_WH set
    warehouse_size = 'MEDIUM'
    auto_suspend = 60
    auto_resume = true
;



-- We need a Snowpark Optimised Warehouse to train the ML model.
-- create or replace warehouse SNOWPARK_OPTIMIZED_WH
--     warehouse_type = 'SNOWPARK-OPTIMIZED'
--     warehouse_size = 'MEDIUM'
--     max_cluster_count = 1
--     min_cluster_count = 1
--     auto_suspend = 60
--     auto_resume = true
--     initially_suspended = true
--     comment = 'Used for memory intensive applications such as training ML models.';
    



-- You shouldn't use ACCOUNTADMIN for creating objects but we're just testing here so it will
-- be fine, we need to make sure ACCOUNTADMIN can use Cortex Functions.
grant database role SNOWFLAKE.CORTEX_USER to role ACCOUNTADMIN;



-- Create DB and Schemas.
create or replace database UCKFIELD_CINEMA;
create or replace schema UCKFIELD_CINEMA.CRM;
create or replace schema UCKFIELD_CINEMA.EPOS;



-- We need some reference data for a number of reasons including using it as an
-- input for our ML models.
use schema UCKFIELD_CINEMA.PUBLIC;



-- Create a reference date table with a date for each ticket combination.
create or replace table UCKFIELD_CINEMA.PUBLIC.DATE_DIM (
    DATE_VALUE date not null,
    YEAR integer not null,
    MONTH integer not null,
    MONTH_NAME varchar not null,
    DAY_OF_MONTH integer not null,
    DAY_OF_WEEK varchar not null,
    DAY_NAME varchar not null,
    WEEK_OF_YEAR integer not null,
    DAY_OF_YEAR integer not null,
    IS_SCHOOL_HOLIDAY_WEEK integer not null,
    TICKET_TYPE varchar not null
) as 
    with CTE_DATE_VALUES as (
        select dateadd( day, seq4(), '2020-01-01') as DATE_VALUE
        -- from table( generator( rowcount => 1825 ) )
        from table( generator( rowcount => 2555 ) )
    )
    select 
        DATE_VALUE,
        year( DATE_VALUE ),
        month( DATE_VALUE ),
        monthname( DATE_VALUE ),
        day( DATE_VALUE ),
        dayofweekiso( DATE_VALUE ),
        dayname( DATE_VALUE ),
        weekofyear( DATE_VALUE ),
        dayofyear( DATE_VALUE ),
        case
            when week( DATE_VALUE ) in ( 8, 9, 14, 15, 16, 17, 22, 23, 31,	32,	33,	34,	35,	36, 44,	45, 51,	52,	53 ) then 1
            else 0
        end as IS_SCHOOL_HOLIDAY_WEEK,
        TICKET_TYPES.*,
        
from CTE_DATE_VALUES

    join ( select * from values('Adult'), ('Child'), ('Senior') ) as TICKET_TYPES
;



-- Create a table listing all the LLMs available to use later on so users can choose which LLM to use.
create or replace table UCKFIELD_CINEMA.PUBLIC.LLM (
    NAME varchar,
    CORTEX_FUNCTION varchar
) as 
    select * from values
        ( 'llama3-8b', 'COMPLETE' ),
        ( 'llama3-70b', 'COMPLETE' ),
        ( 'mistral-large', 'COMPLETE' ),
        ( 'mixtral-8x7b', 'COMPLETE' ),
        ( 'mistral-7b', 'COMPLETE' )
;



-- Now build the sales data.
use schema UCKFIELD_CINEMA.EPOS;



-- Table to hold cinema reference data.
-- create or replace table CINEMAS (
--     CINEMA_ID number,
--     NAME varchar
-- );



-- insert into CINEMAS values
--   ( 1, 'Uckfield' ),
--   ( 2, 'Paris' )
--   ;



-- Table to hold ticket sales data which is what we'll be forecasting using ML functions.
create or replace table TICKET_SALES (
    CINEMA_ID number,
    TICKET_DATE date,
    TICKET_TYPE varchar,
    QTY int,
    TOTAL_AMOUNT number(10,2)
) as 
select
        1 as CINEMA_ID,
        DATE_VALUE as TICKET_DATE,
        TICKET_TYPE,
        case
            when TICKET_TYPE = 'Adult' then uniform( 1250, 1500, random() )
            when TICKET_TYPE = 'Child' then uniform( 800, 900, random() )
            when TICKET_TYPE = 'Senior' then uniform( 700, 750, random() )
        end
        * 
            case
                when IS_SCHOOL_HOLIDAY_WEEK = 1 then 
                    iff( dayofweekiso( DATE_VALUE ) between 1.0 and 5.0, 
                        decode( 
                            TICKET_TYPE,
                            'Adult', 3.0,
                            'Child', 4.0,
                            'Senior', 1.0
                        ), 
                        decode( 
                            TICKET_TYPE,
                            'Adult', 4.0,
                            'Child', 4.0,
                            'Senior', 0.8
                        )
                    )
                else iff( dayofweekiso( DATE_VALUE ) between 1.0 and 5.0, 
                        decode( 
                            TICKET_TYPE,
                            'Adult', 2.0,
                            'Child', 0.5,
                            'Senior', 2.0
                        ), 
                        decode( 
                            TICKET_TYPE,
                            'Adult', 2.5,
                            'Child', 2.0,
                            'Senior', 1.75
                        )
                    )
            end as TOTAL_AMOUNT,
        round( TOTAL_AMOUNT / 
            decode( 
                TICKET_TYPE,
                'Adult', 10.95,
                'Child', 6.95,
                'Senior', 8.00
            )
        , 0 ) as QTY,
        
    from UCKFIELD_CINEMA.PUBLIC.DATE_DIM

    where DATE_VALUE < current_date()

    order by 2
;



-- Build the reviews table.
use schema UCKFIELD_CINEMA.CRM;



-- Create a table to hold the reviews.
create or replace table UCKFIELD_CINEMA.CRM.CUSTOMER_REVIEWS (
    REVIEW_DATE timestamp_ntz,
    NAME varchar,
    EMAIL varchar,
    LANG varchar,
    REVIEW varchar,
    REVIEW_EN varchar,
    SENTIMENT float,
    REVIEW_EN_SUMMARY varchar,
    OFFER varchar
);



-- Insert some data.
set EMAIL_ADDRESS = 'your_email@here.com';

-- Insert reviews including a couple of French ones.
insert overwrite into UCKFIELD_CINEMA.CRM.CUSTOMER_REVIEWS ( REVIEW_DATE, NAME, EMAIL, LANG, REVIEW ) values 
    ( '2024-01-13 09:00:00', 'Dynamo Snare', $EMAIL_ADDRESS, 'en', 'Have been a resident of Uckfield for 30+ years and as such have been a regular patron of the Cinema.  We were due to go to the Lounge tonight, but unfortunately my wife has contracted Covid today.  I rang to see if my tickets could be deferred as it would be irresponsible to attend and was met with a friendly voice, who tried to help. Unfortunately she could not, as we, apparently need to give 24 hours notice to cancel.  I did explain that I wasn''t after a credit, just simply that I could use those tickets on another day.  She put me through to the manager, who although obviously briefed by her colleague just abruptly asked How she could help. Then when a repeated what I said =, she simply just said its company policy and that''s already been explained to you!  So nothing can be done.  I assume I will get a refund for the drinks I''ve ordered ? Or is that company policy to charge for something I haven''t drunk as well as a seat I''ve not sat in?' ),
    
    ( '2024-01-15 09:00:00', 'Chronotrapper', $EMAIL_ADDRESS, 'en', 'First time visiting this cinema, and I was really impressed! The seats are super comfy and I loved being able to order food and drinks to my seat. The sound was set at a comfortable level too, I usually find the sound much too loud at other cinemas. Looking forward to going back!' ),
    
    ( '2024-01-13 09:00:00', 'Atomic Ember', $EMAIL_ADDRESS,  'en','My wife and I randomly found ourselves staying in Uckfield at the weekend (we had to be somewhere local in the morning so stayed over) and we were looking for something to do in the evening. We found this amazing place only a few minutes away from where we were staying. We booked dinner at the restaurant - lovely film inspired decor, a friendly welcome and great service, and really good food, reasonably priced. We then popped over the road to the cinema to watch the new Mission Impossible in their ''lounge'' cinema which was a fantastic experience. Reclining leather armchairs, waitress service before the film for all your food and drink needs - was great to enjoy a pint with the film in such comfort. The only issue is being so comfortable you could end up having a snooze!! If you''re ever passing through I highly recommend stopping here, it''s a great night out. Thanks so much!' ),
    
    ( '2024-01-21 09:00:00', 'Insect Catch', $EMAIL_ADDRESS, 'en', 'If you want just a fantastic cinema experience and a great place to eat beforehand then look no further! The restaurant menu is fantastic. I have celiac disease and their GF menu is brilliant with a large choice. The cinema is just wonderful!! The cinema is clearly loved by those who own it and work there. Forget the normal chain cinemas , Uckfield Picture House offers a truly wonderful and memorable experience. Table service to your seat. Have a beer and food at your seat and enjoy the movie!' ),
    
    ( '2024-01-29 09:00:00', 'Kick Smasher', $EMAIL_ADDRESS,  'en','A lovely little cinema with an easy free car park nearby, which is a plus. Comfy reclining seats, food and drinks brought to you by friendly staff, latest films, what more would you want?!' ),
    
    ( '2024-03-04 09:00:00', 'Mental Vine', $EMAIL_ADDRESS, 'en', 'The Lounge screen is very cool. Comfy large electric reclining seats with raised foot rest (don''t fall asleep!). Get your own mini table with a menu of hot and cold snacks. These aren''t your standards menu offerings you see at cinemas, and the food is tasty. Can order alcohol too. I can recommend the nachos. Someone comes and takes your order from your seat, and brings it to you. Would be better if they had the option to order on your phone from a qr code. Can order in advance though if you book your ticket online.' ),
        
    ( '2024-03-24 09:00:00', 'Steelo', $EMAIL_ADDRESS, 'en', 'We live equi distance between Uckfield & Tunbridge Wells, so have the choice of a large multiplex or the Uckfield picture house. We always pick the lounge at the picture house. Lovely experience. Comfy reclining seats. Table service for drinks and food and great quality viewing experience. We then often pop over to their restaurant over the road which is great quality too.' ),
    
    ( '2024-04-27 09:00:00', 'Tree Raven', $EMAIL_ADDRESS, 'en', 'We were a group of 16 but despite pre-ordering  and paying a deposit we waited nearly two hours for our main courses.  When food came it came all together, hot and very good which was a plus.  Very crowded.' ),
    
    ( '2024-05-02 09:00:00', 'Virus Woman', $EMAIL_ADDRESS, 'en', 'Nice cinema but can be a bit too loud sometimes. Prefer screen 1 to screen 3 so if screen 3 stay away from the farthest seats as only one aisle. Otherwise worth patronising.' ),
    
    ( '2024-05-10 09:00:00', 'Wind Tide', $EMAIL_ADDRESS, 'en', 'I like the Picture House, it''s a lovely cinema and the recent refurbishment is excellent. I''m only giving three stars however as in a recent trip we were treated quite rudely by the management, unnecessarily.' ),
    
    ( '2024-06-13 09:00:00', 'Xavi A', $EMAIL_ADDRESS, 'fr', 'Vraiment super petit cinéma à Uckfield. Ils ont rénové les salles avec moustiquaire afin que même la rangée 1 offre une vue très confortable. Vous pouvez commander de la nourriture et des boissons à apporter à votre place dans l''écran 1 et dans le salon, qui dispose également de sièges inclinables très confortables. Hautement recommandé.' ),
    
    ( '2024-06-19 09:00:00', 'Dancing Cat', $EMAIL_ADDRESS, 'fr', 'Un charmant petit lieu avec un décor intéressant lié au cinéma et servant de la bonne nourriture à des prix raisonnables. Cela vaut vraiment le détour et si vous voulez aussi voir un film, le cinéma est juste de l''autre côté de la rue.' )
;



-- We often end up with lots of tables and views that need to cleared up. This SP
-- will allow you to drop lots of tables or views at once.
create or replace procedure PUBLIC.DROP_TABLES_OR_VIEWS(
    table_type varchar,
    table_database varchar, 
    table_schema varchar, 
    table_pattern varchar
)
returns varchar 
language javascript
execute as caller
as
$$
var table_type = TABLE_TYPE
var table_database = TABLE_DATABASE
var table_schema = TABLE_SCHEMA
var table_pattern = TABLE_PATTERN
var result = "";

var sql_command = `select TABLE_CATALOG || '.' || TABLE_SCHEMA || '.' || TABLE_NAME from ` + table_database + `."INFORMATION_SCHEMA"."TABLES" where TABLE_TYPE = '` + table_type + `'and TABLE_SCHEMA = '` + table_schema + `'  and TABLE_NAME ilike '%` + table_pattern+ `%'`;
var stmt = snowflake.createStatement( {sqlText: sql_command} );
var resultSet = stmt.execute();
while (resultSet.next()){

try { 
    if (table_type == 'BASE TABLE') {  
        var table_name = resultSet.getColumnValue(1);
        var sql_command = `drop table `  + table_name + `;`;    
        snowflake.execute ({sqlText: sql_command});
        result = result + "dropped: " + table_name + "\n"
    }

    else {
        var table_name = resultSet.getColumnValue(1);
        var sql_command = `drop ` + table_type +` ` + table_name + `;`;
        snowflake.execute ({sqlText: sql_command});
        result = result + "dropped: " + table_name + "\n"}
    }
    catch (err)  {
        result =  "Error: " + err.message;
    }
}

return result; 
$$;


-- call drop_tables_or_views ( 'VIEW', 'UCKFIELD_CINEMA', 'EPOS', 'TICKET_SALES');
