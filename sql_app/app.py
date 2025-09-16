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
                    'password': user_data['password']
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
        # Successful login - determine user role and set session state
        if username and (username.startswith('candidate') or username == 'candidate_demo'):
            st.session_state.user_role = "candidate"
            st.session_state.candidate_id = str(uuid.uuid4())[:8]
        elif username == 'interviewer':
            st.session_state.user_role = "interviewer"
        else:
            # Default to candidate for any other users
            st.session_state.user_role = "candidate" 
            st.session_state.candidate_id = str(uuid.uuid4())[:8]
        
        st.session_state.login_time = datetime.now()
        st.rerun()
    

def logout():
    """Handle logout using authenticator"""
    authenticator = get_authenticator()
    authenticator.logout('Logout', 'main')
    
    # Clear custom session state
    keys_to_clear = ['user_role', 'candidate_id', 'login_time']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
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
    
    # Logout button in sidebar
    with st.sidebar:
        st.markdown("### Navigation")
        st.markdown("**Interview Evaluation Tools**")
        # Get fresh authenticator for logout
        authenticator = get_authenticator()
        authenticator.logout('🚪 Logout', 'sidebar')
    
    # Get shared database connection
    conn = get_database_connection()
    
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
            
            # Detailed session analysis
            if selected_session_id:
                session_data = next(s for s in sessions if s[0] == selected_session_id)
                
                st.markdown(f"### 📊 Analysis: Candidate `{selected_session_id}`")
                
                # Session overview metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Queries", session_data[3])
                    st.metric("Success Rate", f"{session_data[8]}%")
                with col2:
                    st.metric("Valid Results", session_data[5])
                    st.metric("Total Points", session_data[6])
                with col3:
                    st.metric("Challenges Tried", session_data[7])
                    duration = session_data[2] - session_data[1]
                    st.metric("Duration", f"{int(duration.total_seconds() / 60)} min")
                with col4:
                    st.metric("Started", session_data[1].strftime('%H:%M'))
                    st.metric("Ended", session_data[2].strftime('%H:%M'))
                
                # Challenge performance breakdown
                challenge_stats = conn.execute("""
                    SELECT 
                        challenge_title,
                        COUNT(*) as attempts,
                        COUNT(CASE WHEN execution_success THEN 1 END) as successful,
                        COUNT(CASE WHEN validation_passed THEN 1 END) as valid,
                        MAX(points_earned) as max_points
                    FROM query_log 
                    WHERE session_id = ? AND challenge_id != 'free_query'
                    GROUP BY challenge_title
                    ORDER BY MAX(timestamp)
                """, [selected_session_id]).fetchall()
                
                if challenge_stats:
                    st.subheader("📈 Challenge Performance")
                    challenge_df = pd.DataFrame(challenge_stats, columns=[
                        'Challenge', 'Attempts', 'Successful', 'Valid', 'Max Points'
                    ])
                    st.dataframe(challenge_df, use_container_width=True)
                
                # Individual query review
                st.subheader("📝 Complete Query History")
                queries = conn.execute("""
                    SELECT 
                        log_id, timestamp, challenge_title, query_text, execution_success, 
                        error_message, result_rows, validation_passed, points_earned
                    FROM query_log 
                    WHERE session_id = ? 
                    ORDER BY timestamp ASC
                """, [selected_session_id]).fetchall()
                
                for i, query in enumerate(queries, 1):
                    # Determine query status styling
                    if query[4]:  # execution success
                        if query[7]:  # validation passed
                            status_icon = "✅"
                        else:
                            status_icon = "⚠️"
                    else:
                        status_icon = "❌"
                    
                    # Create expandable query section
                    header = f"Query #{i}: {query[2]} - {query[1].strftime('%H:%M:%S')} {status_icon}"
                    
                    with st.expander(header, expanded=False):
                        # Query metadata
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(f"**Status:** {'Success' if query[4] else 'Failed'}")
                        with col2:
                            st.markdown(f"**Validation:** {'Valid' if query[7] else 'Invalid'}")
                        with col3:
                            st.markdown(f"**Points:** {query[8]}")
                        with col4:
                            if query[6] is not None:
                                st.markdown(f"**Rows:** {query[6]}")
                        
                        # Query text with syntax highlighting
                        st.markdown("**SQL Query:**")
                        st.code(query[3], language='sql')
                        
                        # Error or success information
                        if not query[4]:  # Failed execution
                            st.error(f"**Error:** {query[5]}")
                        else:
                            if query[6] is not None:
                                st.success(f"Returned {query[6]} rows")
                            if not query[7]:  # Invalid result structure
                                st.warning("Query executed but result structure doesn't match expected columns")
    
    with tab3:
        st.subheader("📤 Export Interview Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Session Summary")
            if st.button("Download Session Summary CSV", use_container_width=True):
                if sessions:
                    sessions_df = pd.DataFrame(sessions, columns=[
                        'Candidate_ID', 'Started', 'Ended', 'Total_Queries', 'Successful', 
                        'Valid_Results', 'Points', 'Challenges', 'Success_Rate'
                    ])
                    csv = sessions_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Summary",
                        data=csv,
                        file_name=f"interview_sessions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
        
        with col2:
            st.markdown("#### 📝 All Queries")
            if st.button("Download All Queries CSV", use_container_width=True):
                all_queries = conn.execute("""
                    SELECT session_id as candidate_id, timestamp, challenge_id, challenge_title, 
                           query_text, execution_success, error_message, 
                           result_rows, validation_passed, points_earned
                    FROM query_log 
                    ORDER BY session_id, timestamp
                """).fetchdf()
                
                if not all_queries.empty:
                    csv = all_queries.to_csv(index=False)
                    st.download_button(
                        label="📥 Download All Queries",
                        data=csv,
                        file_name=f"interview_queries_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )

def show_candidate_interface():
    """Display candidate interface for SQL challenges"""
    st.title("👨‍💻 SQL Proficiency Test")
    st.subheader("Customer Experience Data Engineer Position")
    
    # Initialize database
    conn = get_database_connection()
    
    # Sidebar for navigation and logout
    with st.sidebar:
        st.title("Navigation")
        page = st.selectbox(
            "Choose a section:",
            ["Overview", "SQL Challenges", "Database Schema"]
        )
        
        st.markdown("---")
        st.markdown(f"**Welcome:** {st.session_state.get('name', 'Candidate')}")
        st.markdown(f"**Session ID:** `{st.session_state.candidate_id}`")
        st.markdown("**💡 Quick Tips:**")
        st.markdown("• Schema reference is available in SQL Challenges")
        st.markdown("• Start with Basic challenges")  
        st.markdown("• Practice with the SQL challenges")
        st.markdown("• All queries are automatically logged")
        st.markdown("• Focus on query readability and efficiency")
        
        st.markdown("---")
        # Get fresh authenticator for logout
        authenticator = get_authenticator()
        authenticator.logout('🚪 End Interview', 'sidebar')
    
    # Page handling for candidates
    if page == "Overview":
        st.markdown("""
        ## Welcome to the SQL Proficiency Test
        
        This test is designed to evaluate your SQL skills for a Customer Experience Data Engineer role. 
        You'll work with realistic customer data including:
        
        - **Customer profiles** and account information
        - **Order history** and revenue data
        - **Support tickets** and resolution metrics
        - **Product usage** patterns
        - **Customer satisfaction** surveys
        
        
        ### Instructions:
        1. Review the database schema
        2. Complete the SQL challenges
        3. Aim for clean, efficient, and readable SQL code
        
        **Good luck!** 🚀
        """)
        
    elif page == "Database Schema":
        display_schema()
        
        # Show sample data
        st.subheader("📊 Sample Data Preview")
        table_to_show = st.selectbox(
            "Select table to preview:",
            ['customers', 'orders', 'support_tickets', 'product_usage', 'customer_satisfaction']
        )
        
        sample_data = conn.execute(f"SELECT * FROM {table_to_show} LIMIT 5").fetchdf()
        st.dataframe(sample_data)
        
        st.info(f"**{table_to_show}** has {conn.execute(f'SELECT COUNT(*) FROM {table_to_show}').fetchone()[0]} total rows")
        
    elif page == "SQL Challenges":
        st.subheader("🎯 SQL Challenges")
        
        # Create two columns - schema reference on the left, challenge on the right
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📋 Database Schema Reference")
            
            # Show schema in an expandable sidebar-like format
            schema_info = {
                'customers': {
                    'description': 'Customer information and account details',
                    'columns': [
                        'customer_id (INTEGER)',
                        'email (VARCHAR)', 
                        'signup_date (DATE)',
                        'subscription_tier (VARCHAR)',
                        'company_size (VARCHAR)',
                        'industry (VARCHAR)',
                        'is_active (BOOLEAN)'
                    ]
                },
                'orders': {
                    'description': 'Customer purchase history',
                    'columns': [
                        'order_id (INTEGER)',
                        'customer_id (INTEGER)',
                        'order_date (DATE)',
                        'amount (DECIMAL)',
                        'status (VARCHAR)',
                        'product_category (VARCHAR)'
                    ]
                },
                'support_tickets': {
                    'description': 'Customer support interactions',
                    'columns': [
                        'ticket_id (INTEGER)',
                        'customer_id (INTEGER)',
                        'created_date (DATETIME)',
                        'category (VARCHAR)',
                        'priority (VARCHAR)',
                        'status (VARCHAR)',
                        'resolution_time_hours (INTEGER)',
                        'satisfaction_score (INTEGER)'
                    ]
                },
                'product_usage': {
                    'description': 'Daily product usage metrics',
                    'columns': [
                        'customer_id (INTEGER)',
                        'usage_date (DATE)',
                        'sessions (INTEGER)',
                        'page_views (INTEGER)',
                        'time_spent_minutes (INTEGER)',
                        'feature_used (VARCHAR)'
                    ]
                },
                'customer_satisfaction': {
                    'description': 'Customer satisfaction survey responses',
                    'columns': [
                        'survey_id (INTEGER)',
                        'customer_id (INTEGER)',
                        'survey_date (DATE)',
                        'nps_score (INTEGER)',
                        'product_satisfaction (INTEGER)',
                        'support_satisfaction (INTEGER)',
                        'likelihood_to_recommend (INTEGER)',
                        'overall_satisfaction (INTEGER)'
                    ]
                }
            }
            
            for table_name, table_info in schema_info.items():
                with st.expander(f"**{table_name.upper()}**", expanded=True):
                    st.caption(table_info['description'])
                    for column in table_info['columns']:
                        st.text(f"• {column}")
        
        with col2:
            challenges = get_sql_challenges()
            
            # Challenge selection
            selected_challenge = st.selectbox(
                "Select a challenge:",
                challenges,
                format_func=lambda x: f"{x['title']}"
            )
        
            st.markdown(f"### {selected_challenge['title']}")
            st.markdown(f"**Description:** {selected_challenge['description']}")
            
            if st.button("💡 Show Hint"):
                st.info(f"**Hint:** {selected_challenge['hint']}")
            
            st.markdown("**Expected columns:**")
            st.write(", ".join(selected_challenge['expected_columns']))
            
            # SQL Editor
            st.markdown("### Write your SQL query:")
            user_query = st_ace(
                language='sql',
                theme='monokai',
                key=f"sql_editor_{selected_challenge['id']}",
                height=200,
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
                            challenge_id=selected_challenge['id'],
                            challenge_title=selected_challenge['title'],
                            query_text=user_query,
                            execution_success=False,
                            error_message=error,
                            result_rows=0,
                            validation_passed=False,
                            points_earned=0
                        )
                    else:
                        st.success("✅ Query executed successfully!")
                        
                        # Validate result structure
                        is_valid, validation_message = validate_query_result(result, selected_challenge['expected_columns'])
                        result_rows = len(result) if result is not None else 0
                        
                        if is_valid:
                            st.success(f"🎉 {validation_message}")
                            points_earned = selected_challenge['points']
                        else:
                            st.warning(f"⚠️ {validation_message}")
                            points_earned = 0
                        
                        # Log successful query
                        log_query(
                            conn=conn,
                            challenge_id=selected_challenge['id'],
                            challenge_title=selected_challenge['title'],
                            query_text=user_query,
                            execution_success=True,
                            error_message=None,
                            result_rows=result_rows,
                            validation_passed=is_valid,
                            points_earned=points_earned
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
            # Determine role based on username (fallback)
            username = st.session_state.get("username", "")
            if username == 'interviewer':
                st.session_state.user_role = "interviewer"
            else:
                st.session_state.user_role = "candidate"
                st.session_state.candidate_id = str(uuid.uuid4())[:8]
        
        # Route based on user role
        if st.session_state.user_role == "candidate":
            show_candidate_interface()
        elif st.session_state.user_role == "interviewer":
            show_interviewer_interface()
        else:
            st.error("Invalid user role. Please login again.")
            # Get fresh authenticator for logout
            authenticator = get_authenticator()
            authenticator.logout('Logout', 'main')
    else:
        show_login_page()

if __name__ == "__main__":
    main()
