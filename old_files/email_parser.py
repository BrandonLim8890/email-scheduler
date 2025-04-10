import os, sys, email
import pandas as pd
import re
import hashlib
from tqdm import tqdm
from collections import Counter

print("Loading dataset...")
emails_df = pd.read_csv("emails.csv")
print(f"Original dataset size: {emails_df.shape}")


def get_text_from_email(msg):
    """Extract text content from email objects"""
    parts = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                parts.append(part.get_payload())
    else:
        parts.append(msg.get_payload())
    return "".join(parts)


def contains_meeting_keywords(text, subject):
    """Simple meeting keyword detection"""
    if not isinstance(text, str) or not isinstance(subject, str):
        return False

    keywords = [
        "meet",
        "meeting",
        "schedule",
        "calendar",
        "appointment",
        "call",
        "conference",
        "agenda",
        "invite",
        "invitation",
        "available",
        "availability",
        "reschedule",
        "cancel",
        "tomorrow",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "9am",
        "10am",
        "11am",
        "12pm",
        "1pm",
        "2pm",
        "3pm",
        "4pm",
        "5pm",
        "let's discuss",
        "available to talk",
        "available to meet",
        "lunch",
        "breakfast",
        "coffee",
    ]

    text_lower = text.lower()
    subject_lower = subject.lower()

    for keyword in keywords:
        if keyword in subject_lower:
            return True

    for keyword in keywords:
        if keyword in text_lower:
            return True

    return False


def is_transcript_or_newsletter(text, subject):
    """Check if the email looks like a transcript or newsletter"""
    if not isinstance(text, str) or not isinstance(subject, str):
        return False

    if len(text) > 2000:
        transcript_markers = ["transcript", "meeting minutes", "earnings call"]
        text_lower = text.lower()
        for marker in transcript_markers:
            if marker in text_lower:
                return True

    party_keywords = ["party", "celebration", "happy hour", "holiday party"]
    if any(keyword in subject.lower() for keyword in party_keywords):
        return False

    return False


print("Sampling emails...")
sample_size = min(200_000, len(emails_df))
sampled_emails = emails_df.sample(sample_size)
print(f"Sample size: {len(sampled_emails)}")

sampled_emails["user"] = sampled_emails["file"].apply(lambda x: x.split("/")[0])

user_counts = sampled_emails["user"].value_counts()
print("\nTop 10 users by email count:")
print(user_counts.head(10))

top_users = user_counts.head(4).index.tolist()
print(f"\nSelected top 4 users: {top_users}")

target_email_count = 3000

for user in top_users:
    print(f"\nProcessing emails for {user}...")

    user_emails = sampled_emails[sampled_emails["user"] == user].copy()
    print(f"Found {len(user_emails)} emails for {user}")

    batch_size = 500
    all_processed = []
    seen_fingerprints = set()

    for start_idx in tqdm(range(0, len(user_emails), batch_size)):
        end_idx = min(start_idx + batch_size, len(user_emails))
        batch = user_emails.iloc[start_idx:end_idx].copy()

        processed_batch = []
        for _, row in batch.iterrows():
            if len(all_processed) >= target_email_count:
                break

            try:
                msg = email.message_from_string(row["message"])

                from_value = msg.get("From", "")
                to_value = msg.get("To", "")
                subject = msg.get("Subject", "")
                content = get_text_from_email(msg)

                if "sent" in row["file"].lower():
                    continue

                if is_transcript_or_newsletter(content, subject):
                    continue

                fingerprint = hashlib.md5(
                    (from_value + subject + content[:100]).encode()
                ).hexdigest()

                if fingerprint in seen_fingerprints:
                    continue

                seen_fingerprints.add(fingerprint)

                is_meeting = contains_meeting_keywords(content, subject)

                email_data = {
                    "file": row["file"],
                    "user": user,
                    "From": from_value,
                    "To": to_value,
                    "Subject": subject,
                    "Date": msg.get("Date", ""),
                    "content": content,
                    "meeting_related": is_meeting,
                }

                processed_batch.append(email_data)

            except Exception as e:
                print(f"Error processing email: {e}")
                continue

        all_processed.extend(processed_batch)

        if len(all_processed) >= target_email_count:
            print(
                f"Reached target of {target_email_count} emails. Stopping processing."
            )
            break

    processed_df = pd.DataFrame(all_processed)

    if len(processed_df) == 0:
        print(f"No valid emails found for {user}. Skipping.")
        continue

    if len(processed_df) > target_email_count:
        meeting_emails = processed_df[processed_df["meeting_related"]]
        non_meeting_emails = processed_df[~processed_df["meeting_related"]]

        meeting_count = len(meeting_emails)

        non_meeting_to_keep = min(
            target_email_count - meeting_count, len(non_meeting_emails)
        )

        if non_meeting_to_keep > 0:
            kept_non_meeting = non_meeting_emails.sample(non_meeting_to_keep)
            processed_df = pd.concat([meeting_emails, kept_non_meeting])
        else:
            processed_df = meeting_emails

    meeting_count = processed_df["meeting_related"].sum()
    total_count = len(processed_df)
    print(f"Final dataset size: {total_count}")
    print(
        f"Meeting-related emails: {meeting_count} ({meeting_count/total_count*100:.1f}%)"
    )

    processed_df["formatted_email"] = processed_df.apply(
        lambda row: (
            f"From: {row.get('From', 'Unknown')}\n"
            f"To: {row.get('To', 'Unknown')}\n"
            f"Date: {row.get('Date', 'Unknown')}\n"
            f"Subject: {row.get('Subject', '(No Subject)')}\n\n"
            f"{row.get('content', '')}"
        ),
        axis=1,
    )

    csv_filename = f"{user}_simple_inbox.csv"
    processed_df.to_csv(csv_filename, index=False)

    txt_filename = f"{user}_simple_inbox.txt"
    with open(txt_filename, "w") as f:
        f.write(f"INBOX EMAILS FOR: {user}\n")
        f.write("=" * 50 + "\n\n")

        f.write("MEETING-RELATED EMAILS\n")
        f.write("-" * 25 + "\n\n")

        meeting_emails = processed_df[processed_df["meeting_related"]]
        for _, row in meeting_emails.iterrows():
            f.write(f"FROM: {row.get('From', 'Unknown')}\n")
            f.write(f"TO: {row.get('To', 'Unknown')}\n")
            f.write(f"DATE: {row.get('Date', 'Unknown')}\n")
            f.write(f"SUBJECT: {row.get('Subject', '(No Subject)')}\n\n")
            f.write(f"{row.get('content', '')}\n\n")
            f.write("-" * 50 + "\n\n")

        f.write("\nOTHER EMAILS\n")
        f.write("-" * 25 + "\n\n")

        non_meeting_emails = processed_df[~processed_df["meeting_related"]]
        for _, row in non_meeting_emails.iterrows():
            f.write(f"FROM: {row.get('From', 'Unknown')}\n")
            f.write(f"TO: {row.get('To', 'Unknown')}\n")
            f.write(f"DATE: {row.get('Date', 'Unknown')}\n")
            f.write(f"SUBJECT: {row.get('Subject', '(No Subject)')}\n\n")
            f.write(f"{row.get('content', '')}\n\n")
            f.write("-" * 50 + "\n\n")

    print(f"Created {csv_filename} and {txt_filename}")

print("\nDone! Created simple inbox datasets for the top 4 most active users.")
