CREATE SCHEMA IF NOT EXISTS findociq;

CREATE TABLE IF NOT EXISTS findociq.documents (
    application_id text NOT NULL,
    document_id text NOT NULL,
    sha256 char(64) NOT NULL,
    state text NOT NULL CHECK (state IN ('active', 'quarantined', 'retention_pending', 'deleted')),
    policy_hash char(64) NOT NULL,
    retained_until timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (application_id, document_id)
);

CREATE INDEX IF NOT EXISTS findociq_documents_retention_idx
    ON findociq.documents (state, retained_until);

CREATE TABLE IF NOT EXISTS findociq.jwt_replay (
    jti text PRIMARY KEY,
    expires_at timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS findociq.deletion_receipts (
    application_id text PRIMARY KEY,
    policy_hash char(64) NOT NULL,
    artifact_hashes jsonb NOT NULL,
    deleted_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS findociq.key_rotation_events (
    event_id uuid PRIMARY KEY,
    old_key_version text NOT NULL,
    new_key_version text NOT NULL,
    object_count integer NOT NULL CHECK (object_count >= 0),
    policy_hash char(64) NOT NULL,
    occurred_at timestamptz NOT NULL DEFAULT now()
);
