import os
from supabase import create_client, Client
import pandas as pd
from df import monitoring, clean_data


clean_data(monitoring())
