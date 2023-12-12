# basics
import numpy as np
import pandas as pd
import getpass
import json
import os

# Zooniverse tools
from panoptes_client import Panoptes, Workflow
from panoptes_aggregation.extractors.utilities import annotation_by_task
from panoptes_aggregation.extractors import question_extractor
from panoptes_aggregation.reducers import question_consensus_reducer



def download_classifications(WORKFLOW_ID, client):
    """
    Downloads data from Zooniverse

    Args:
        WORKFLOW_ID (int): Workflow ID of workflow being aggregated
        client: Logged in Zooniverse client

    Returns:
        classification_data (DataFrame): Raw classifications from Zooniverse
    """

    workflow = Workflow(WORKFLOW_ID)

    # generate the classifications 
    with client:
        classification_export = workflow.get_export('classifications', generate=True, wait=True)
        classification_rows = [row for row in tqdm(classification_export.csv_dictreader())]

    # convert to pandas dataframe
    classification_data = pd.DataFrame.from_dict(classification_rows)

    return classification_data



def extract_data(classification_data):
    """
    Extracts annotations from the classification data

    Args:
        classification_data (DataFrame): Raw classifications from Zooniverse

    Returns:
        extracted_data (DataFrame): Extracted annotations from raw classification data
    """
    # set up our list where we will store the extracted data temporarily
    extracted_rows = []

    # iterate through our classification data
    for i in range(len(classification_data)):

        # access the specific row and extract the annotations
        row = classification_data.iloc[i]
        for annotation in json.loads(row.annotations):

            row_annotation = annotation_by_task({'annotations': [annotation]})
            extract = question_extractor(row_annotation)

            # add the extracted annotations to our temporary list along with some other additional data
            extracted_rows.append({
                'classification_id': row.classification_id,
                'subject_id':        row.subject_ids,
                'user_name':         row.user_name,
                'user_id':           row.user_id,
                'created_at':        row.created_at,
                'data':              json.dumps(extract),
                'task':              annotation['task']
            })


    # convert the extracted data to a pandas dataframe and sort
    extracted_data = pd.DataFrame.from_dict(extracted_rows)
    extracted_data.sort_values(['subject_id', 'created_at'], inplace=True)

    return extracted_data



def last_filter(data):
    """
    Determines the most recently submitted classifications
    """
    last_time = data.created_at.max()
    ldx = data.created_at == last_time
    return data[ldx]


def aggregate_data(extracted_data):
    """
    Aggregates question data from extracted annotations

    Args:
        extracted_data (DataFrame): Extracted annotations from raw classifications

    Returns:
        aggregated_data (DataFrame): Aggregated data for the given workflow
    """
    # generate an array of unique subject ids - these are the ones that we will iterate over
    subject_ids_unique = np.unique(extracted_data.subject_id)

    # set up a temporary list to store reduced data
    aggregated_rows = []

    # determine the total number of tasks
    tasks = np.unique(extracted_data.task)

    # iterating over each unique subject id
    for i in range(len(subject_ids_unique)):

        # determine the subject_id to work on
        subject_id = subject_ids_unique[i]

        # filter the extract_data dataframe for only the subject_id that is being worked on
        extract_data_subject = extracted_data[extracted_data.subject_id==subject_id].drop_duplicates()

        for task in tasks:

            extract_data_filtered = extract_data_subject[extract_data_subject.task == task]

            # if there are less unique user submissions than classifications, filter for the most recently updated classification
            if (len(extract_data_filtered.user_name.unique()) < len(extract_data_filtered)):
                extract_data_filtered = extract_data_filtered.groupby(['user_name'], group_keys=False).apply(last_filter)

            # iterate through the filtered extract data to prepare for the reducer
            classifications_to_reduce = [json.loads(extract_data_filtered.iloc[j].data) for j in range(len(extract_data_filtered))]

            # use the Zooniverse question_consesus_reducer to get the final consensus
            reduction = question_consensus_reducer(classifications_to_reduce)

            # add the subject id to our reduction data
            reduction['subject_id'] = subject_id
            reduction['task'] = task

            # add the data to our temporary list
            aggregated_rows.append(reduction)


    # converting the result to a dataframe
    aggregated_data = pd.DataFrame.from_dict(aggregated_rows)

    # drop rows that are nan
    aggregated_data.dropna(inplace=True)

    return aggregated_data





def batch_aggregation(generate_new_classifications=False, WORKFLOW_ID=13193): 
    """
    Downloads raw classifications, extracts annotations, and aggregates data

    Args:
        WORKFLOW_ID (int): Workflow ID of workflow being aggregated
        client: Logged in Zooniverse client

    Returns:
        aggregated_data (DataFrame): Aggregated data for the given workflow
    """

    if generate_new_classifications:
        # connect to client and download data
        print('Sign in to zooniverse.org:')
        client = Panoptes.client(username=getpass.getpass('username: '), password=getpass.getpass('password: '))
        print('Generating classification data - could take some time')
        classification_data = get_data_from_zooniverse(WORKFLOW_ID=WORKFLOW_ID, client=client)
        print('Saving classifications')
        classification_data.to_csv('superluminous-supernovae-classifications.csv', index=False)
    else:
        # or just open the file
        print('Loading classifications')
        classification_data = pd.read_csv('superluminous-supernovae-classifications.csv')

    # limit classifications to those for the relevant workflow
    classification_data = classification_data[classification_data.workflow_id==WORKFLOW_ID]

    # extract annotations
    print('Extracting annotations')
    extracted_data = extract_data(classification_data=classification_data)

    # aggregate annotations
    print('Aggregating data')
    final_data = aggregate_data(extracted_data=extracted_data)

    return final_data







