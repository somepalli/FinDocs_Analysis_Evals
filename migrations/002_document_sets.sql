CREATE TABLE IF NOT EXISTS findociq.document_sets (
    set_id TEXT PRIMARY KEY,
    application_id TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS document_sets_application_idx
    ON findociq.document_sets(application_id);
