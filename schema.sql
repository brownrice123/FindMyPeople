CREATE TABLE IF NOT EXISTS users (
    username TEXT,
    home_city TEXT,
    collected_at TEXT,
    PRIMARY KEY (username, home_city)
);

CREATE TABLE IF NOT EXISTS activity (
    username TEXT,
    subreddit TEXT,
    comment_count INTEGER,
    PRIMARY KEY (username, subreddit)
);

-- Aggregation function used by the embeddings layer (Session 6)
CREATE OR REPLACE FUNCTION get_top_subreddits(n INT)
RETURNS TABLE(subreddit TEXT) LANGUAGE sql AS $$
    SELECT subreddit
    FROM activity
    GROUP BY subreddit
    ORDER BY COUNT(*) DESC
    LIMIT n;
$$;