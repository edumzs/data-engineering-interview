import streamlit as st
import duckdb
import pandas as pd
from datetime import datetime, timedelta
import random
import numpy as np
import uuid
import os
import time
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_ace import st_ace

# Page configuration
st.set_page_config(
    page_title="SQL Proficiency Test - Customer Experience Data Engineer",
    page_icon="🔍",
    layout="wide"
)

# Load authentication configuration
@st.cache_data
def load_config():
    """Load authentication configuration from Streamlit secrets or fallback to YAML file"""
    try:
        # Try to load from Streamlit secrets first (for deployment)
        # Convert secrets to regular dict to avoid recursion issues
        credentials = {}
        if 'credentials' in st.secrets:
            credentials = {
                'usernames': {}
            }
            for username in st.secrets['credentials']['usernames']:
                user_data = st.secrets['credentials']['usernames'][username]
                credentials['usernames'][username] = {
                    'email': user_data['email'],
                    'name': user_data['name'],
                    'password': user_data['password'],
                    'role': user_data.get('role', 'candidate')  # Default to candidate if no role
                }
        
        cookie = {}
        if 'cookie' in st.secrets:
            cookie = {
                'expiry_days': st.secrets['cookie']['expiry_days'],
                'key': st.secrets['cookie']['key'],
                'name': st.secrets['cookie']['name']
            }
        
        config = {
            'credentials': credentials,
            'cookie': cookie
        }
        return config
    except (KeyError, AttributeError):
        # Fallback to local YAML file for development
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path) as file:
                config = yaml.load(file, Loader=SafeLoader)
            return config
        else:
            st.error("No authentication configuration found. Please set up secrets or config.yaml file.")
            st.stop()

# Initialize authenticator
def get_user_role_and_setup_session(username):
    """Get user role from config and setup session state"""
    config = load_config()
    user_role = config['credentials']['usernames'].get(username, {}).get('role', 'candidate')
    st.session_state.user_role = user_role
    
    # Set candidate ID if user is a candidate
    if user_role == "candidate":
        st.session_state.candidate_id = str(uuid.uuid4())[:8]
    
    return user_role

def get_authenticator():
    """Get a fresh authenticator instance"""
    config = load_config()
    
    return stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

# Authentication and role management
def show_login_page():
    """Display professional login page using Streamlit Authenticator"""
    st.title("🔍 SQL Proficiency Test")
    st.subheader("Customer Experience Data Engineer Position")
    
    # Create login widget
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Please Login")
        # Get fresh authenticator for login
        authenticator = get_authenticator()
        authenticator.login(location="main")
        
        # Get authentication status from session state
        name = st.session_state.get("name")
        authentication_status = st.session_state.get("authentication_status")
        username = st.session_state.get("username")
    
    # Handle authentication results
    if authentication_status == False:
        st.error('❌ Username/password is incorrect')
    elif authentication_status == None:
        st.warning('⚠️ Please enter your username and password')
        
    
    elif authentication_status:
        # Successful login - determine user role from configuration
        get_user_role_and_setup_session(username)
        st.session_state.login_time = datetime.now()
        st.rerun()
    

def handle_logout(button_text='Logout', location='main'):
    """Handle logout with customizable button text and location"""
    authenticator = get_authenticator()
    authenticator.logout(button_text, location)
    
    # Clear custom session state
    keys_to_clear = ['user_role', 'candidate_id', 'login_time']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

def logout():
    """Handle logout using authenticator"""
    handle_logout('Logout', 'main')
    
    st.rerun()

# Initialize DuckDB connection - SIMPLIFIED APPROACH
@st.cache_resource
def init_shared_database():
    """Initialize a single shared DuckDB instance for all sessions"""
    # Use in-memory database to avoid file locking issues
    conn = duckdb.connect(':memory:')
    
    # Create sample data
    create_sample_data(conn)
    
    # Create query log table for persistence
    create_query_log_table(conn)
    
    return conn

def get_database_connection():
    """Get the shared database connection"""
    return init_shared_database()

def create_query_log_table(conn):
    """Create table to store submitted queries for persistence"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_log (
            log_id INTEGER PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            challenge_id VARCHAR,
            challenge_title VARCHAR,
            query_text TEXT,
            execution_success BOOLEAN,
            error_message TEXT,
            result_rows INTEGER,
            validation_passed BOOLEAN,
            points_earned INTEGER,
            session_id VARCHAR
        )
    """)

def create_sample_data(conn):
    """Create comprehensive sample data for customer experience testing"""
    
    # Check if customers table already exists
    existing_tables = conn.execute("SHOW TABLES").fetchall()
    table_names = [table[0] for table in existing_tables]
    
    if 'customers' in table_names:
        # Data already exists, skip creation
        return
    
    # Customers table
    customers_data = []
    for i in range(1, 1001):
        signup_date = datetime.now() - timedelta(days=random.randint(30, 730))
        customers_data.append({
            'customer_id': i,
            'email': f'customer_{i}@example.com',
            'signup_date': signup_date,
            'subscription_tier': random.choice(['basic', 'pro', 'enterprise']),
            'company_size': random.choice(['1-10', '11-50', '51-200', '201-1000', '1000+']),
            'industry': random.choice(['education', 'healthcare', 'technology', 'finance', 'retail']),
            'is_active': np.random.choice([True, False], p=[0.8, 0.2])
        })
    
    customers_df = pd.DataFrame(customers_data)
    conn.register('customers', customers_df)
    
    # Orders table
    orders_data = []
    order_id = 1
    for customer_id in range(1, 1001):
        num_orders = random.randint(0, 10)
        customer_signup = customers_df[customers_df['customer_id'] == customer_id]['signup_date'].iloc[0]
        
        for _ in range(num_orders):
            order_date = customer_signup + timedelta(days=random.randint(0, 365))
            orders_data.append({
                'order_id': order_id,
                'customer_id': customer_id,
                'order_date': order_date,
                'amount': round(random.uniform(50, 2000), 2),
                'status': np.random.choice(['completed', 'cancelled', 'refunded'], p=[0.8, 0.1, 0.1]),
                'product_category': random.choice(['course', 'coaching', 'community', 'website'])
            })
            order_id += 1
    
    orders_df = pd.DataFrame(orders_data)
    conn.register('orders', orders_df)
    
    # Support tickets table
    tickets_data = []
    ticket_id = 1
    for customer_id in range(1, 1001):
        num_tickets = random.randint(0, 5)
        customer_signup = customers_df[customers_df['customer_id'] == customer_id]['signup_date'].iloc[0]
        
        for _ in range(num_tickets):
            created_date = customer_signup + timedelta(days=random.randint(0, 365))
            resolution_time = random.randint(1, 168)  # 1 hour to 1 week
            tickets_data.append({
                'ticket_id': ticket_id,
                'customer_id': customer_id,
                'created_date': created_date,
                'category': random.choice(['technical', 'billing', 'feature_request', 'general']),
                'priority': np.random.choice(['low', 'medium', 'high'], p=[0.5, 0.3, 0.2]),
                'status': np.random.choice(['open', 'in_progress', 'resolved', 'closed'], p=[0.1, 0.15, 0.65, 0.1]),
                'resolution_time_hours': resolution_time if random.random() > 0.2 else None,
                'satisfaction_score': random.randint(1, 5) if random.random() > 0.3 else None
            })
            ticket_id += 1
    
    tickets_df = pd.DataFrame(tickets_data)
    conn.register('support_tickets', tickets_df)
    
    # Product usage table
    usage_data = []
    for customer_id in range(1, 1001):
        if random.random() > 0.2:  # 80% of customers have usage data
            days_active = random.randint(30, 365)
            for day in range(days_active):
                if random.random() > 0.3:  # Not every day has usage
                    usage_date = datetime.now() - timedelta(days=day)
                    usage_data.append({
                        'customer_id': customer_id,
                        'usage_date': usage_date,
                        'sessions': random.randint(0, 10),
                        'page_views': random.randint(0, 50),
                        'time_spent_minutes': random.randint(0, 240),
                        'feature_used': random.choice(['dashboard', 'reports', 'settings', 'content_creation', 'analytics'])
                    })
    
    usage_df = pd.DataFrame(usage_data)
    conn.register('product_usage', usage_df)
    
    # Customer satisfaction surveys
    satisfaction_data = []
    survey_id = 1
    for customer_id in range(1, 1001):
        if random.random() > 0.6:  # 40% response rate
            survey_date = datetime.now() - timedelta(days=random.randint(0, 90))
            satisfaction_data.append({
                'survey_id': survey_id,
                'customer_id': customer_id,
                'survey_date': survey_date,
                'nps_score': random.randint(0, 10),
                'product_satisfaction': random.randint(1, 5),
                'support_satisfaction': random.randint(1, 5),
                'likelihood_to_recommend': random.randint(1, 5),
                'overall_satisfaction': random.randint(1, 5)
            })
            survey_id += 1
    
    satisfaction_df = pd.DataFrame(satisfaction_data)
    conn.register('customer_satisfaction', satisfaction_df)

def display_schema():
    """Display database schema for reference"""
    schema_info = {
        'customers': {
            'description': 'Customer information and account details',
            'columns': [
                'customer_id (INTEGER): Unique customer identifier',
                'email (VARCHAR): Customer email address',
                'signup_date (DATE): Date customer signed up',
                'subscription_tier (VARCHAR): basic, pro, or enterprise',
                'company_size (VARCHAR): Company size category',
                'industry (VARCHAR): Customer industry',
                'is_active (BOOLEAN): Whether customer is currently active'
            ]
        },
        'orders': {
            'description': 'Customer purchase history',
            'columns': [
                'order_id (INTEGER): Unique order identifier',
                'customer_id (INTEGER): Foreign key to customers',
                'order_date (DATE): Date order was placed',
                'amount (DECIMAL): Order amount in USD',
                'status (VARCHAR): completed, cancelled, or refunded',
                'product_category (VARCHAR): Type of product purchased'
            ]
        },
        'support_tickets': {
            'description': 'Customer support interactions',
            'columns': [
                'ticket_id (INTEGER): Unique ticket identifier',
                'customer_id (INTEGER): Foreign key to customers',
                'created_date (DATETIME): When ticket was created',
                'category (VARCHAR): technical, billing, feature_request, or general',
                'priority (VARCHAR): low, medium, or high',
                'status (VARCHAR): open, in_progress, resolved, or closed',
                'resolution_time_hours (INTEGER): Time to resolve (NULL if unresolved)',
                'satisfaction_score (INTEGER): 1-5 rating (NULL if not provided)'
            ]
        },
        'product_usage': {
            'description': 'Daily product usage metrics',
            'columns': [
                'customer_id (INTEGER): Foreign key to customers',
                'usage_date (DATE): Date of usage',
                'sessions (INTEGER): Number of user sessions',
                'page_views (INTEGER): Number of page views',
                'time_spent_minutes (INTEGER): Total time spent in product',
                'feature_used (VARCHAR): Primary feature used that day'
            ]
        },
        'customer_satisfaction': {
            'description': 'Customer satisfaction survey responses',
            'columns': [
                'survey_id (INTEGER): Unique survey response identifier',
                'customer_id (INTEGER): Foreign key to customers',
                'survey_date (DATE): Date survey was completed',
                'nps_score (INTEGER): Net Promoter Score (0-10)',
                'product_satisfaction (INTEGER): Product satisfaction (1-5)',
                'support_satisfaction (INTEGER): Support satisfaction (1-5)',
                'likelihood_to_recommend (INTEGER): Likelihood to recommend (1-5)',
                'overall_satisfaction (INTEGER): Overall satisfaction (1-5)'
            ]
        }
    }
    
    st.subheader("📋 Database Schema")
    for table_name, table_info in schema_info.items():
        with st.expander(f"**{table_name.upper()}** - {table_info['description']}"):
            for column in table_info['columns']:
                st.write(f"• {column}")

def get_sql_challenges():
    """Define SQL challenges for customer experience data engineering"""
    return [
        {
            'id': 'basic_1',
            'title': 'Customer Overview',
            'difficulty': 'Basic',
            'description': 'Write a query to show the total number of customers by subscription tier and industry.',
            'expected_columns': ['subscription_tier', 'industry', 'customer_count'],
            'points': 10,
            'hint': 'Use GROUP BY with multiple columns and COUNT()'
        },
        {
            'id': 'basic_2',
            'title': 'Revenue Analysis',
            'difficulty': 'Basic',
            'description': 'Calculate the total revenue and average order value for each product category.',
            'expected_columns': ['product_category', 'total_revenue', 'avg_order_value'],
            'points': 10,
            'hint': 'Use SUM() and AVG() with GROUP BY'
        },
        {
            'id': 'intermediate_1',
            'title': 'Customer Lifetime Value',
            'difficulty': 'Intermediate',
            'description': 'Calculate the total revenue per customer and identify the top 10 customers by lifetime value. Include customer email and signup date.',
            'expected_columns': ['customer_id', 'email', 'signup_date', 'lifetime_value'],
            'points': 20,
            'hint': 'JOIN customers with orders and use SUM() with LIMIT'
        },
        {
            'id': 'intermediate_2',
            'title': 'Support Ticket Metrics',
            'difficulty': 'Intermediate',
            'description': 'Create a summary showing average resolution time and satisfaction score by ticket category and priority.',
            'expected_columns': ['category', 'priority', 'avg_resolution_time_hours', 'avg_satisfaction_score'],
            'points': 20,
            'hint': 'Use AVG() on non-null values and GROUP BY multiple columns'
        },
        {
            'id': 'advanced_1',
            'title': 'Churn Risk Analysis',
            'difficulty': 'Advanced',
            'description': 'Identify customers who haven\'t placed an order in the last 90 days but are still active. Include their last order date and total historical spend.',
            'expected_columns': ['customer_id', 'email', 'last_order_date', 'total_spent', 'days_since_last_order'],
            'points': 30,
            'hint': 'Use window functions, date arithmetic, and complex joins'
        },
        {
            'id': 'advanced_2',
            'title': 'Customer Journey Analysis',
            'difficulty': 'Advanced',
            'description': 'Create a cohort analysis showing customer retention by signup month. Show the percentage of customers who made at least one purchase in each subsequent month after signup.',
            'expected_columns': ['signup_month', 'month_number', 'customers_in_cohort', 'customers_active', 'retention_rate'],
            'points': 30,
            'hint': 'Use date functions, window functions, and cohort analysis techniques'
        },
        {
            'id': 'expert_1',
            'title': 'Product Engagement Score',
            'difficulty': 'Expert',
            'description': 'Calculate a comprehensive engagement score for each customer based on usage metrics, support interactions, and satisfaction scores. Rank customers by engagement level.',
            'expected_columns': ['customer_id', 'engagement_score', 'usage_score', 'support_score', 'satisfaction_score', 'engagement_rank'],
            'points': 40,
            'hint': 'Combine multiple metrics with weighted scoring and ranking functions'
        }
    ]

def execute_query(conn, query):
    """Execute SQL query and return results"""
    try:
        result = conn.execute(query).fetchdf()
        return result, None
    except Exception as e:
        return None, str(e)

def log_query(conn, challenge_id, challenge_title, query_text, execution_success, error_message, result_rows, validation_passed, points_earned):
    """Log submitted query to database for persistence"""
    
    # Use candidate_id for session tracking
    candidate_id = st.session_state.get('candidate_id', 'unknown')
    
    try:
        # Get next log_id
        max_id_result = conn.execute("SELECT COALESCE(MAX(log_id), 0) + 1 FROM query_log").fetchone()
        next_id = max_id_result[0] if max_id_result else 1
        
        conn.execute("""
            INSERT INTO query_log 
            (log_id, timestamp, challenge_id, challenge_title, query_text, execution_success, error_message, result_rows, validation_passed, points_earned, session_id)
            VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            next_id,
            challenge_id,
            challenge_title, 
            query_text,
            execution_success,
            error_message,
            result_rows,
            validation_passed,
            points_earned,
            candidate_id
        ])
    except Exception as e:
        st.error(f"Error logging query: {str(e)}")

def validate_query_result(result, expected_columns):
    """Validate query result against expected structure"""
    if result is None or result.empty:
        return False, "Query returned no results"
    
    result_columns = set(result.columns.str.lower())
    expected_columns_set = set([col.lower() for col in expected_columns])
    
    if not expected_columns_set.issubset(result_columns):
        missing_columns = expected_columns_set - result_columns
        return False, f"Missing expected columns: {', '.join(missing_columns)}"
    
    return True, "Query structure is correct"

def show_interviewer_interface():
    """Display interviewer interface for query review and analysis"""
    st.title("👩‍🏫 Interviewer Dashboard")
    st.subheader(f"Welcome, {st.session_state.get('name', 'Interviewer')}")
    
    # Navigation sidebar
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.selectbox(
            "Choose a section:",
            ["Query Review", "Upload Data"]
        )
        st.markdown("---")
        # Logout button
        handle_logout('🚪 Logout', 'sidebar')
    
    # Get shared database connection
    conn = get_database_connection()
    
    if page == "Query Review":
        show_query_review_section(conn)
    elif page == "Upload Data":
        show_upload_data_section(conn)

def show_query_review_section(conn):
    """Display query review and analysis section"""
    # Check if there are any query logs
    try:
        total_logs = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
    except:
        st.error("Query log table not found. No interviews have been conducted yet.")
        return
    
    if total_logs == 0:
        st.info("📭 No candidate submissions found yet.")
        st.markdown("Candidates need to complete interviews first.")
        return
    
    st.success(f"📊 Found {total_logs} query submissions across all candidates")
    
    # Get all sessions with summary stats
    sessions = conn.execute("""
        SELECT 
            session_id,
            MIN(timestamp) as first_query,
            MAX(timestamp) as last_query,
            COUNT(*) as total_queries,
            COUNT(CASE WHEN execution_success THEN 1 END) as successful_queries,
            COUNT(CASE WHEN validation_passed THEN 1 END) as valid_queries,
            SUM(points_earned) as total_points,
            COUNT(DISTINCT challenge_id) as challenges_attempted,
            ROUND(AVG(CASE WHEN execution_success THEN 1.0 ELSE 0.0 END) * 100, 1) as success_rate
        FROM query_log 
        GROUP BY session_id 
        ORDER BY first_query DESC
    """).fetchall()
    
    # Navigation tabs
    tab1, tab2, tab3 = st.tabs(["📋 Sessions Overview", "🔍 Detailed Review", "📤 Export Data"])
    
    with tab1:
        st.subheader("📋 All Interview Sessions")
        
        if sessions:
            # Display session summary table
            sessions_df = pd.DataFrame(sessions, columns=[
                'Candidate ID', 'Started', 'Ended', 'Total Queries', 'Successful', 
                'Valid Results', 'Points', 'Challenges', 'Success Rate %'
            ])
            st.dataframe(sessions_df, use_container_width=True)
            
            # Quick stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Candidates", len(sessions))
            with col2:
                avg_queries = sum(s[3] for s in sessions) / len(sessions)
                st.metric("Avg Queries/Candidate", f"{avg_queries:.1f}")
            with col3:
                avg_points = sum(s[6] for s in sessions) / len(sessions)
                st.metric("Avg Points", f"{avg_points:.1f}")
            with col4:
                avg_success = sum(s[8] for s in sessions) / len(sessions)
                st.metric("Avg Success Rate", f"{avg_success:.1f}%")
        else:
            st.info("No interview sessions found yet.")
    
    with tab2:
        st.subheader("🔍 Candidate Deep Dive")
        
        if sessions:
            # Session selector
            session_options = []
            for session in sessions:
                duration = session[2] - session[1]
                duration_mins = int(duration.total_seconds() / 60)
                option = f"{session[0]} - {session[3]} queries, {session[6]} pts, {duration_mins}m ({session[1].strftime('%Y-%m-%d %H:%M')})"
                session_options.append(option)
            
            selected_session_display = st.selectbox("Select candidate for detailed review:", session_options)
            selected_session_id = selected_session_display.split(' - ')[0]
            
            # Detailed session analysis would go here
            st.info("Detailed session analysis coming soon...")
        else:
            st.info("No sessions available for detailed review.")
    
    with tab3:
        st.subheader("📤 Export Data")
        st.info("Export functionality coming soon...")

def show_upload_data_section(conn):
    """Display CSV upload section for interviewers"""
    st.subheader("📤 Upload Data")
    st.markdown("Upload CSV files to create new tables in the database for interview scenarios.")
    
    # Alert about CSV requirements
    st.info("📋 **Important:** CSV files must have column headers in the first row for proper table creation.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with headers in the first row"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            
            # Display preview
            st.subheader("📋 Data Preview")
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Rows:** {len(df)}")
            st.write(f"**Columns:** {len(df.columns)}")
            
            # Show first few rows
            st.dataframe(df.head(10))
            
            # Table name input
            table_name = st.text_input(
                "Table Name",
                value=uploaded_file.name.replace('.csv', '').lower().replace(' ', '_'),
                help="Enter a name for the new table (will be created in DuckDB)"
            )
            
            # Validate table name
            if table_name and not table_name.replace('_', '').isalnum():
                st.error("Table name should only contain letters, numbers, and underscores")
                return
            
            if st.button("🚀 Create Table"):
                if table_name:
                    try:
                        # Check if table already exists
                        existing_tables = conn.execute("SHOW TABLES").fetchall()
                        table_names = [table[0] for table in existing_tables]
                        
                        if table_name in table_names:
                            if st.checkbox(f"Table '{table_name}' already exists. Replace it?"):
                                conn.execute(f"DROP TABLE {table_name}")
                            else:
                                st.warning(f"Table '{table_name}' already exists. Check the box above to replace it.")
                                return
                        
                        # Create table from DataFrame
                        conn.register(table_name, df)
                        
                        # Verify table creation
                        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                        
                        st.success(f"✅ Table '{table_name}' created successfully with {row_count} rows!")
                        
                        # Show table schema
                        st.subheader("📊 Table Schema")
                        schema_info = conn.execute(f"DESCRIBE {table_name}").fetchdf()
                        st.dataframe(schema_info)
                        
                        # Show sample data from new table
                        st.subheader("🔍 Sample Data")
                        sample_data = conn.execute(f"SELECT * FROM {table_name} LIMIT 5").fetchdf()
                        st.dataframe(sample_data)
                        
                    except Exception as e:
                        st.error(f"❌ Error creating table: {str(e)}")
                else:
                    st.warning("Please enter a table name")
                    
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.info("Make sure your CSV file has headers in the first row and is properly formatted.")
    
    # Show existing tables
    st.subheader("📋 Current Database Tables")
    try:
        existing_tables = conn.execute("SHOW TABLES").fetchall()
        if existing_tables:
            table_names = [table[0] for table in existing_tables]
            
            # Create expandable sections for each table
            for table_name in table_names:
                # Check if this table should be expanded (due to drop confirmation, edit mode, or success message)
                confirm_key = f"confirm_drop_{table_name}"
                edit_key = f"edit_schema_{table_name}"
                success_key = f"desc_saved_{table_name}"
                is_expanded = (confirm_key in st.session_state and st.session_state[confirm_key]) or \
                             (edit_key in st.session_state and st.session_state[edit_key]) or \
                             (success_key in st.session_state and st.session_state[success_key])
                
                with st.expander(f"**{table_name.upper()}**", expanded=is_expanded):
                    try:
                        # Get row count
                        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                        st.write(f"**Rows:** {row_count}")
                        
                        # Get schema and add descriptions
                        schema = conn.execute(f"DESCRIBE {table_name}").fetchdf()
                        
                        # Add description column
                        desc_key = f"descriptions_{table_name}"
                        descriptions = st.session_state.get(desc_key, {})
                        
                        # Add default descriptions for built-in tables
                        default_table_descriptions = {
                            'query_log': {
                                'log_id': 'Unique identifier for each query submission',
                                'timestamp': 'When the query was submitted',
                                'challenge_id': 'Identifier of the SQL challenge',
                                'challenge_title': 'Title of the SQL challenge',
                                'query_text': 'The SQL query submitted by candidate',
                                'execution_success': 'Whether the query executed without errors',
                                'error_message': 'Error details if query failed',
                                'result_rows': 'Number of rows returned by the query',
                                'validation_passed': 'Whether query result matches expected structure',
                                'points_earned': 'Points awarded for the query',
                                'session_id': 'Candidate session identifier'
                            },
                            'customers': {
                                'customer_id': 'Unique customer identifier',
                                'email': 'Customer email address',
                                'signup_date': 'Date customer signed up',
                                'subscription_tier': 'basic, pro, or enterprise',
                                'company_size': 'Company size category',
                                'industry': 'Customer industry',
                                'is_active': 'Whether customer is currently active'
                            },
                            'orders': {
                                'order_id': 'Unique order identifier',
                                'customer_id': 'Foreign key to customers',
                                'order_date': 'Date order was placed',
                                'amount': 'Order amount in USD',
                                'status': 'completed, cancelled, or refunded',
                                'product_category': 'Type of product purchased'
                            },
                            'support_tickets': {
                                'ticket_id': 'Unique ticket identifier',
                                'customer_id': 'Foreign key to customers',
                                'created_date': 'When ticket was created',
                                'category': 'technical, billing, feature_request, or general',
                                'priority': 'low, medium, or high',
                                'status': 'open, in_progress, resolved, or closed',
                                'resolution_time_hours': 'Time to resolve (NULL if unresolved)',
                                'satisfaction_score': '1-5 rating (NULL if not provided)'
                            },
                            'product_usage': {
                                'customer_id': 'Foreign key to customers',
                                'usage_date': 'Date of usage',
                                'sessions': 'Number of user sessions',
                                'page_views': 'Number of page views',
                                'time_spent_minutes': 'Total time spent in product',
                                'feature_used': 'Primary feature used that day'
                            },
                            'customer_satisfaction': {
                                'survey_id': 'Unique survey response identifier',
                                'customer_id': 'Foreign key to customers',
                                'survey_date': 'Date survey was completed',
                                'nps_score': 'Net Promoter Score (0-10)',
                                'product_satisfaction': 'Product satisfaction (1-5)',
                                'support_satisfaction': 'Support satisfaction (1-5)',
                                'likelihood_to_recommend': 'Likelihood to recommend (1-5)',
                                'overall_satisfaction': 'Overall satisfaction (1-5)'
                            }
                        }
                        
                        # Get default descriptions for this table if available (case-insensitive)
                        table_name_lower = table_name.lower()
                        if table_name_lower in default_table_descriptions:
                            default_descriptions = default_table_descriptions[table_name_lower]
                            # Merge with any custom descriptions, giving priority to custom ones
                            descriptions = {**default_descriptions, **descriptions}
                        else:
                            # Debug: show which table name we're looking for
                            st.write(f"Debug: Looking for '{table_name}' (lowercase: '{table_name_lower}') in default descriptions")
                            st.write(f"Available keys: {list(default_table_descriptions.keys())}")
                        
                        # Create description column
                        schema['description'] = schema['column_name'].apply(
                            lambda col: descriptions.get(col, "")
                        )
                        
                        st.write("**Schema:**")
                        st.dataframe(schema, use_container_width=True)
                        
                        # Show sample data, edit schema, and drop table buttons
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button(f"Show sample data", key=f"sample_{table_name}"):
                                sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 5").fetchdf()
                                st.dataframe(sample, use_container_width=True)
                        
                        with col2:
                            # Don't allow editing the query_log table
                            if table_name != 'query_log':
                                if st.button(f"✏️ Edit schema", key=f"edit_{table_name}", type="secondary"):
                                    # Use session state to track edit mode
                                    edit_key = f"edit_schema_{table_name}"
                                    if edit_key not in st.session_state:
                                        st.session_state[edit_key] = False
                                    
                                    if not st.session_state[edit_key]:
                                        st.session_state[edit_key] = True
                                        st.rerun()
                        
                        with col3:
                            # Don't allow dropping the query_log table
                            if table_name != 'query_log':
                                if st.button(f"🗑️ Drop table", key=f"drop_{table_name}", type="secondary"):
                                    # Use session state to track confirmation
                                    confirm_key = f"confirm_drop_{table_name}"
                                    if confirm_key not in st.session_state:
                                        st.session_state[confirm_key] = False
                                    
                                    if not st.session_state[confirm_key]:
                                        st.session_state[confirm_key] = True
                                        st.rerun()
                        
                        # Show schema editing interface if edit was clicked
                        edit_key = f"edit_schema_{table_name}"
                        if edit_key in st.session_state and st.session_state[edit_key]:
                            st.markdown("### ✏️ Edit Table Schema")
                            st.info("Add descriptions for each column to help candidates understand the data structure.")
                            
                            # Initialize or get existing descriptions
                            desc_key = f"descriptions_{table_name}"
                            if desc_key not in st.session_state:
                                st.session_state[desc_key] = {}
                            
                            # Get current descriptions including defaults
                            current_descriptions = st.session_state.get(desc_key, {})
                            
                            # Add default descriptions for built-in tables if not already set
                            table_name_lower = table_name.lower()
                            if table_name_lower in default_table_descriptions:
                                default_descriptions = default_table_descriptions[table_name_lower]
                                # Only add defaults for columns that don't have custom descriptions
                                for col, desc in default_descriptions.items():
                                    if col not in current_descriptions:
                                        current_descriptions[col] = desc
                                st.session_state[desc_key] = current_descriptions
                            
                            # Get current schema
                            schema = conn.execute(f"DESCRIBE {table_name}").fetchdf()
                            
                            # Create input fields for each column description
                            for _, row in schema.iterrows():
                                column_name = row['column_name']
                                column_type = row['column_type']
                                
                                current_desc = current_descriptions.get(column_name, "")
                                new_desc = st.text_input(
                                    f"**{column_name}** ({column_type})",
                                    value=current_desc,
                                    key=f"desc_{table_name}_{column_name}",
                                    placeholder="Enter column description..."
                                )
                                st.session_state[desc_key][column_name] = new_desc
                            
                            # Save and Cancel buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"💾 Save descriptions", key=f"save_desc_{table_name}", type="primary"):
                                    # Store descriptions in a persistent way (using session state for now)
                                    # In a real implementation, you might want to store this in a database
                                    
                                    # Set success message flag and clear edit mode
                                    success_key = f"desc_saved_{table_name}"
                                    st.session_state[success_key] = True
                                    del st.session_state[edit_key]
                                    st.rerun()
                            with col2:
                                if st.button(f"❌ Cancel", key=f"cancel_edit_{table_name}"):
                                    del st.session_state[edit_key]
                                    st.rerun()
                        
                        # Show success message if descriptions were just saved
                        success_key = f"desc_saved_{table_name}"
                        if success_key in st.session_state and st.session_state[success_key]:
                            st.success(f"✅ Descriptions saved for '{table_name}'!")
                            # Clear the success flag after showing it
                            del st.session_state[success_key]
                        
                        # Show confirmation dialog if drop was clicked
                        confirm_key = f"confirm_drop_{table_name}"
                        if confirm_key in st.session_state and st.session_state[confirm_key]:
                            st.warning(f"⚠️ Are you sure you want to drop table '{table_name}'? This action cannot be undone.")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button(f"✅ Yes, drop it", key=f"confirm_yes_{table_name}", type="primary"):
                                    try:
                                        # Try dropping as table first, then as view if that fails
                                        try:
                                            conn.execute(f"DROP TABLE {table_name}")
                                            st.success(f"✅ Table '{table_name}' dropped successfully!")
                                        except Exception:
                                            # If table drop fails, try dropping as view
                                            conn.execute(f"DROP VIEW {table_name}")
                                            st.success(f"✅ View '{table_name}' dropped successfully!")
                                        
                                        # Clear confirmation state and any descriptions
                                        del st.session_state[confirm_key]
                                        desc_key = f"descriptions_{table_name}"
                                        if desc_key in st.session_state:
                                            del st.session_state[desc_key]
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"❌ Error dropping object: {str(e)}")
                            with col2:
                                if st.button(f"❌ Cancel", key=f"confirm_no_{table_name}"):
                                    # Clear confirmation state
                                    del st.session_state[confirm_key]
                                    st.rerun()
                            
                    except Exception as e:
                        st.error(f"Error accessing table {table_name}: {str(e)}")
        else:
            st.info("No tables found in database")
    except Exception as e:
        st.error(f"Error listing tables: {str(e)}")

def show_candidate_interface():
    """Display candidate interface for SQL challenges"""
    st.title("👨‍💻 SQL Proficiency Test")
    st.subheader("Customer Experience Data Engineer Position")
    
    # Initialize database
    conn = get_database_connection()
    
    # Sidebar for navigation and logout
    with st.sidebar:
        st.title("Navigation")
        # Direct to SQL Challenges - no navigation needed
        page = "SQL Challenges"
        
        st.markdown("---")
        st.markdown(f"**Welcome:** {st.session_state.get('name', 'Candidate')}")
        st.markdown(f"**Session ID:** `{st.session_state.candidate_id}`")
        
        st.markdown("---")
        # End interview button
        handle_logout('🚪 End Interview', 'sidebar')
    
    # Direct to SQL Challenges
    if page == "SQL Challenges":
        st.subheader("🎯 SQL Challenges")
        
        # Database Schema Reference (full width)
        st.subheader("📋 Database Schema Reference")
        
        # Get all available tables dynamically
        try:
            existing_tables = conn.execute("SHOW TABLES").fetchall()
            if existing_tables:
                table_names = [table[0] for table in existing_tables if table[0] != 'query_log']
                
                for table_name in table_names:
                    try:
                        # Get table schema
                        schema = conn.execute(f"DESCRIBE {table_name}").fetchdf()
                        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                        
                        with st.expander(f"**{table_name.upper()}** ({row_count} rows)", expanded=False):
                            st.markdown("**Schema:**")
                            
                            # Check if there are saved descriptions for this table
                            desc_key = f"descriptions_{table_name}"
                            descriptions = st.session_state.get(desc_key, {})
                            
                            # Add default descriptions for built-in tables (same as interviewer interface)
                            default_table_descriptions = {
                                'query_log': {
                                    'log_id': 'Unique identifier for each query submission',
                                    'timestamp': 'When the query was submitted',
                                    'challenge_id': 'Identifier of the SQL challenge',
                                    'challenge_title': 'Title of the SQL challenge',
                                    'query_text': 'The SQL query submitted by candidate',
                                    'execution_success': 'Whether the query executed without errors',
                                    'error_message': 'Error details if query failed',
                                    'result_rows': 'Number of rows returned by the query',
                                    'validation_passed': 'Whether query result matches expected structure',
                                    'points_earned': 'Points awarded for the query',
                                    'session_id': 'Candidate session identifier'
                                },
                                'customers': {
                                    'customer_id': 'Unique customer identifier',
                                    'email': 'Customer email address',
                                    'signup_date': 'Date customer signed up',
                                    'subscription_tier': 'basic, pro, or enterprise',
                                    'company_size': 'Company size category',
                                    'industry': 'Customer industry',
                                    'is_active': 'Whether customer is currently active'
                                },
                                'orders': {
                                    'order_id': 'Unique order identifier',
                                    'customer_id': 'Foreign key to customers',
                                    'order_date': 'Date order was placed',
                                    'amount': 'Order amount in USD',
                                    'status': 'completed, cancelled, or refunded',
                                    'product_category': 'Type of product purchased'
                                },
                                'support_tickets': {
                                    'ticket_id': 'Unique ticket identifier',
                                    'customer_id': 'Foreign key to customers',
                                    'created_date': 'When ticket was created',
                                    'category': 'technical, billing, feature_request, or general',
                                    'priority': 'low, medium, or high',
                                    'status': 'open, in_progress, resolved, or closed',
                                    'resolution_time_hours': 'Time to resolve (NULL if unresolved)',
                                    'satisfaction_score': '1-5 rating (NULL if not provided)'
                                },
                                'product_usage': {
                                    'customer_id': 'Foreign key to customers',
                                    'usage_date': 'Date of usage',
                                    'sessions': 'Number of user sessions',
                                    'page_views': 'Number of page views',
                                    'time_spent_minutes': 'Total time spent in product',
                                    'feature_used': 'Primary feature used that day'
                                },
                                'customer_satisfaction': {
                                    'survey_id': 'Unique survey response identifier',
                                    'customer_id': 'Foreign key to customers',
                                    'survey_date': 'Date survey was completed',
                                    'nps_score': 'Net Promoter Score (0-10)',
                                    'product_satisfaction': 'Product satisfaction (1-5)',
                                    'support_satisfaction': 'Support satisfaction (1-5)',
                                    'likelihood_to_recommend': 'Likelihood to recommend (1-5)',
                                    'overall_satisfaction': 'Overall satisfaction (1-5)'
                                }
                            }
                            
                            # Get default descriptions for this table if available (case-insensitive)
                            table_name_lower = table_name.lower()
                            if table_name_lower in default_table_descriptions:
                                default_descriptions = default_table_descriptions[table_name_lower]
                                # Merge with any custom descriptions, giving priority to custom ones
                                descriptions = {**default_descriptions, **descriptions}
                            
                            for _, row in schema.iterrows():
                                column_name = row['column_name']
                                column_type = row['column_type']
                                nullable = "NULL" if row['null'] == 'YES' else "NOT NULL"
                                
                                # Show column with description if available
                                if column_name in descriptions and descriptions[column_name].strip():
                                    st.text(f"• {column_name} ({column_type}): {descriptions[column_name]}")
                                else:
                                    st.text(f"• {column_name} ({column_type}) - {nullable}")
                            
                            # Show sample data
                            st.markdown("**Sample Data:**")
                            sample_data = conn.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchdf()
                            st.dataframe(sample_data, use_container_width=True)
                            
                    except Exception as e:
                        with st.expander(f"**{table_name.upper()}**", expanded=False):
                            st.error(f"Error accessing table: {str(e)}")
            else:
                st.info("No tables found in database. Upload data through the interviewer interface first.")
        except Exception as e:
            st.error(f"Error retrieving database schema: {str(e)}")
        
        # SQL Query Editor (full width, below schema)
        st.markdown("### Write your SQL query:")
        user_query = st_ace(
            language='sql',
            theme='monokai',
            key="sql_editor_custom",
            height=300,
            placeholder="-- Write your SQL query here\nSELECT ... FROM ..."
        )
        
        if st.button("▶️ Execute Query"):
            if user_query.strip():
                result, error = execute_query(conn, user_query)
                
                if error:
                    st.error(f"❌ Query Error: {error}")
                    # Log failed query
                    log_query(
                        conn=conn,
                        challenge_id="custom_query",
                        challenge_title="Custom SQL Query",
                        query_text=user_query,
                        execution_success=False,
                        error_message=error,
                        result_rows=0,
                        validation_passed=False,
                        points_earned=0
                    )
                else:
                    st.success("✅ Query executed successfully!")
                    result_rows = len(result) if result is not None else 0
                    
                    # Log successful query
                    log_query(
                        conn=conn,
                        challenge_id="custom_query",
                        challenge_title="Custom SQL Query",
                        query_text=user_query,
                        execution_success=True,
                        error_message=None,
                        result_rows=result_rows,
                        validation_passed=True,
                        points_earned=0
                    )
                    
                    # Display results
                    st.markdown("### Query Results:")
                    st.dataframe(result)
                    st.info(f"Returned {len(result)} rows")
            else:
                st.warning("Please enter a SQL query")
                

def main():
    """Main application entry point with role-based routing"""
    
    # Check authentication status using streamlit-authenticator session state
    authentication_status = st.session_state.get("authentication_status")
    
    if authentication_status == False:
        st.error('❌ Username/password is incorrect')
        show_login_page()
        return
    elif authentication_status == None:
        show_login_page()
        return
    elif authentication_status == True:
        # User is authenticated, check if role is set
        if 'user_role' not in st.session_state:
            # Get role from configuration based on username
            username = st.session_state.get("username", "")
            get_user_role_and_setup_session(username)
        
        # Route based on user role
        if st.session_state.user_role == "candidate":
            show_candidate_interface()
        elif st.session_state.user_role == "interviewer":
            show_interviewer_interface()
        else:
            st.error("Invalid user role. Please login again.")
            # Handle logout for invalid role
            handle_logout('Logout', 'main')
    else:
        show_login_page()

if __name__ == "__main__":
    main()
