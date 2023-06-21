CREATE TABLE IF NOT EXISTS user_info (
  id            INT AUTO_INCREMENT PRIMARY KEY,
  date          DATE        NOT NULL,
  guild_name    VARCHAR(20) NOT NULL,
  name          VARCHAR(20) NOT NULL,
  weekly_point  INT         NOT NULL,
  channel_point INT         NOT NULL,
  flag_point    INT         NOT NULL,
  update_at     DATETIME    NOT NULL,

  UNIQUE KEY (guild_name, name, date),
  INDEX      idx_date (date),
  INDEX      idx_name (name),
  INDEX      idx_guild_name (guild_name)
);
